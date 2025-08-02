use crate::{CubeRuntime, element::CubeElement, ops::numeric::empty_device, tensor::CubeTensor};
use burn_tensor::ops::{ScanConfig, ScanOp};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

/// Dimension-aware scan kernel
/// Each thread processes one scan line along the specified dimension
#[cube(launch_unchecked)]
pub fn scan_dim_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    #[comptime] scan_dim: u32,
    #[comptime] operation: ScanOp,
) {
    let thread_id = ABSOLUTE_POS;
    
    let scan_dim_size = input.shape(scan_dim);
    let total_elements = input.len();
    let num_scan_lines = total_elements / scan_dim_size;
    
    if thread_id >= num_scan_lines {
        terminate!();
    }
    
    // Calculate the base index for this scan line
    // This is similar to how repeat_dim calculates offsets
    let mut base_offset = 0u32;
    let mut temp_thread_id = thread_id;
    
    for dim in 0..input.rank() {
        if dim != scan_dim {
            let stride = input.stride(dim);
            let size = input.shape(dim);
            let idx = temp_thread_id % size;
            base_offset += idx * stride;
            temp_thread_id /= size;
        }
    }
    
    // Now scan along the specified dimension
    let stride = input.stride(scan_dim);
    
    if scan_dim_size > 0 {
        // Initialize first element of scan line
        let first_idx = base_offset;
        output[first_idx] = input[first_idx];
        
        // Perform scan operation along the dimension
        for i in 1..scan_dim_size {
            let current_idx = base_offset + i * stride;
            let prev_idx = base_offset + (i - 1) * stride;
            
            match operation {
                ScanOp::Add => {
                    output[current_idx] = output[prev_idx] + input[current_idx];
                }
                ScanOp::Mul => {
                    output[current_idx] = output[prev_idx] * input[current_idx];
                }
                ScanOp::Max => {
                    let prev_val = output[prev_idx];
                    let curr_val = input[current_idx];
                    output[current_idx] = F::max(prev_val, curr_val);
                }
                ScanOp::Min => {
                    let prev_val = output[prev_idx];
                    let curr_val = input[current_idx];
                    output[current_idx] = F::min(prev_val, curr_val);
                }
                ScanOp::And | ScanOp::Or | ScanOp::Xor => {
                    // Logical operations not supported for float tensors
                    output[current_idx] = input[current_idx];
                }
            }
        }
    }
}

/// Launch dimension-aware scan kernel for GPU execution
pub fn gpu_scan_serial<R: CubeRuntime, E: CubeElement>(
    tensor: CubeTensor<R>,
    config: ScanConfig,
) -> CubeTensor<R> {
    // Create output tensor with same shape as input
    let output = empty_device::<R, E>(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape.clone(),
    );
    
    let scan_dim = config.dim;
    let scan_dim_size = tensor.shape.dims[scan_dim];
    let total_elements = tensor.shape.num_elements();
    let num_scan_lines = total_elements / scan_dim_size;
    
    // Set up kernel launch parameters
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_scan_lines, cube_dim);
    
    // Launch kernel based on element type
    match E::dtype() {
        burn_tensor::DType::F32 => {
            let input_f32 = tensor.as_tensor_arg::<f32>(1);
            let output_f32 = output.as_tensor_arg::<f32>(1);
            
            unsafe {
                scan_dim_kernel::launch_unchecked::<f32, R>(
                    &tensor.client,
                    cube_count,
                    cube_dim,
                    input_f32,
                    output_f32,
                    scan_dim as u32,
                    config.op,
                );
            }
        }
        burn_tensor::DType::F64 => {
            let input_f64 = tensor.as_tensor_arg::<f64>(1);
            let output_f64 = output.as_tensor_arg::<f64>(1);
            
            unsafe {
                scan_dim_kernel::launch_unchecked::<f64, R>(
                    &tensor.client,
                    cube_count,
                    cube_dim,
                    input_f64,
                    output_f64,
                    scan_dim as u32,
                    config.op,
                );
            }
        }
        _ => panic!("Unsupported data type for GPU scan: {:?}", E::dtype()),
    }
    
    output
}

/// Main GPU scan function - uses parallel implementation for small scan dimensions
pub fn gpu_scan<R: CubeRuntime, E: CubeElement>(
    tensor: CubeTensor<R>,
    config: ScanConfig,
) -> CubeTensor<R> {
    let scan_dim_size = tensor.shape.dims[config.dim];
    
    // Use parallel implementation for scan dimensions that fit in a single workgroup
    if scan_dim_size <= 256 && matches!(E::dtype(), burn_tensor::DType::F32 | burn_tensor::DType::F64) {
        gpu_scan_parallel::<R, E>(tensor, config)
    } else {
        // Fall back to serial implementation for larger scan dimensions
        gpu_scan_serial::<R, E>(tensor, config)
    }
}

/// Parallel GPU scan implementation using workgroup-level parallelism
/// Now properly handles multi-dimensional tensors with dimension-aware scanning
pub fn gpu_scan_parallel<R: CubeRuntime, E: CubeElement>(
    tensor: CubeTensor<R>,
    config: ScanConfig,
) -> CubeTensor<R> {
    let output = empty_device::<R, E>(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape.clone(),
    );
    
    let scan_dim_size = tensor.shape.dims[config.dim];
    
    // Only use parallel scan if the scan dimension fits in workgroup size
    if scan_dim_size <= 256 {
        let total_elements = tensor.shape.num_elements();
        let num_scan_lines = total_elements / scan_dim_size;
        
        // Each workgroup processes one scan line, with scan_dim_size threads per workgroup
        let cube_dim = CubeDim::new(scan_dim_size.min(256) as u32, 1, 1);
        let cube_count = CubeCount::Static(num_scan_lines as u32, 1, 1);
        
        match E::dtype() {
            burn_tensor::DType::F32 => {
                let input_f32 = tensor.as_tensor_arg::<f32>(1);
                let output_f32 = output.as_tensor_arg::<f32>(1);
                
                unsafe {
                    scan_parallel_dim_kernel::launch_unchecked::<f32, R>(
                        &tensor.client,
                        cube_count,
                        cube_dim,
                        input_f32,
                        output_f32,
                        config.dim as u32,
                        config.op,
                    );
                }
            }
            burn_tensor::DType::F64 => {
                let input_f64 = tensor.as_tensor_arg::<f64>(1);
                let output_f64 = output.as_tensor_arg::<f64>(1);
                
                unsafe {
                    scan_parallel_dim_kernel::launch_unchecked::<f64, R>(
                        &tensor.client,
                        cube_count,
                        cube_dim,
                        input_f64,
                        output_f64,
                        config.dim as u32,
                        config.op,
                    );
                }
            }
            _ => panic!("Unsupported data type for parallel GPU scan: {:?}", E::dtype()),
        }
        
        output
    } else {
        // Fall back to serial implementation for large scan dimensions
        gpu_scan_serial::<R, E>(tensor, config)
    }
}

/// Parallel scan kernel using workgroup-level parallelism for dimension-aware scanning
/// Each workgroup processes one scan line along the specified dimension
#[cube(launch_unchecked)]
pub fn scan_parallel_dim_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    #[comptime] scan_dim: u32,
    #[comptime] operation: ScanOp,
) {
    let workgroup_id = CUBE_POS_X;
    let thread_id = UNIT_POS_X;
    
    let scan_dim_size = input.shape(scan_dim);
    let total_elements = input.len();
    let num_scan_lines = total_elements / scan_dim_size;
    
    if workgroup_id >= num_scan_lines {
        terminate!();
    }
    
    // Calculate the base index for this scan line using the same logic as serial version
    let mut base_offset = 0u32;
    let mut temp_workgroup_id = workgroup_id;
    
    for dim in 0..input.rank() {
        if dim != scan_dim {
            let stride = input.stride(dim);
            let size = input.shape(dim);
            let idx = temp_workgroup_id % size;
            base_offset += idx * stride;
            temp_workgroup_id /= size;
        }
    }
    
    let stride = input.stride(scan_dim);
    
    // Use shared memory for parallel scan within the scan line
    let mut shared_data = SharedMemory::<F>::new(256);
    
    // Load input data into shared memory
    if thread_id < scan_dim_size {
        let input_idx = base_offset + thread_id * stride;
        shared_data[thread_id] = input[input_idx];
    } else {
        // Initialize out-of-bounds elements with identity values
        shared_data[thread_id] = match operation {
            ScanOp::Add => F::new(0.0),
            ScanOp::Mul => F::new(1.0),
            ScanOp::Max => F::new(-1000000.0), // Large negative value
            ScanOp::Min => F::new(1000000.0),  // Large positive value
            _ => F::new(0.0),
        };
    }
    sync_cube();
    
    // Hillis-Steele parallel scan algorithm
    let mut step = 1u32;
    while step < 256 && step < scan_dim_size {
        if thread_id >= step {
            let left_val = shared_data[thread_id - step];
            let current_val = shared_data[thread_id];
            shared_data[thread_id] = match operation {
                ScanOp::Add => left_val + current_val,
                ScanOp::Mul => left_val * current_val,
                ScanOp::Max => F::max(left_val, current_val),
                ScanOp::Min => F::min(left_val, current_val),
                _ => current_val,
            };
        }
        sync_cube();
        step *= 2;
    }
    
    // Write results back to output
    if thread_id < scan_dim_size {
        let output_idx = base_offset + thread_id * stride;
        output[output_idx] = shared_data[thread_id];
    }
}
