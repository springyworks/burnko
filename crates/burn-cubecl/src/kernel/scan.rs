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
    
    // Handle empty tensors
    if total_elements == 0 || scan_dim_size == 0 {
        return output;
    }
    
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

/// Main GPU scan function - always uses GPU parallel implementation
/// No automatic switching to CPU or serial implementations
pub fn gpu_scan<R: CubeRuntime, E: CubeElement>(
    tensor: CubeTensor<R>,
    config: ScanConfig,
) -> CubeTensor<R> {
    // Always use GPU parallel implementation - no crossover switching
    // This ensures consistent GPU execution within compute graphs
    gpu_scan_parallel::<R, E>(tensor, config)
}


/// Parallel scan kernel using workgroup-level parallelism for dimension-aware scanning
/// Enhanced to handle larger scan dimensions efficiently
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
    
    let cube_size = CUBE_DIM_X;
    
    // Handle arrays efficiently based on their size relative to cube size
    if scan_dim_size <= cube_size {
        // Small arrays: Use efficient Hillis-Steele parallel scan
        if thread_id < scan_dim_size {
            let element_idx = base_offset + thread_id * stride;
            output[element_idx] = input[element_idx];
        }
        
        sync_cube();
        
        // Hillis-Steele parallel prefix sum
        let mut step = 1u32;
        while step < scan_dim_size {
            if thread_id >= step && thread_id < scan_dim_size {
                let current_idx = base_offset + thread_id * stride;
                let prev_idx = base_offset + (thread_id - step) * stride;
                
                let prev_val = output[prev_idx];
                let curr_val = output[current_idx];
                
                let new_val = match operation {
                    ScanOp::Add => prev_val + curr_val,
                    ScanOp::Mul => prev_val * curr_val,
                    ScanOp::Max => {
                        if prev_val > curr_val { prev_val } else { curr_val }
                    }
                    ScanOp::Min => {
                        if prev_val < curr_val { prev_val } else { curr_val }
                    }
                    _ => curr_val,
                };
                
                output[current_idx] = new_val;
            }
            
            step *= 2;
            sync_cube();
        }
    } else {
        // Large arrays: Simple parallel chunked approach - ALL THREADS WORK!
        let chunk_size = (scan_dim_size + cube_size - 1) / cube_size;
        
        // Step 1: Each thread handles its chunk - copy input to output first
        for i in 0..chunk_size {
            let idx = thread_id * chunk_size + i;
            if idx < scan_dim_size {
                let element_idx = base_offset + idx * stride;
                output[element_idx] = input[element_idx];
            }
        }
        
        sync_cube();
        
        // Step 2: Each thread does local scan on its chunk
        for i in 1..chunk_size {
            let idx = thread_id * chunk_size + i;
            if idx < scan_dim_size {
                let current_idx = base_offset + idx * stride;
                let prev_idx = base_offset + (idx - 1) * stride;
                
                let prev_val = output[prev_idx];
                let curr_val = output[current_idx];
                
                output[current_idx] = match operation {
                    ScanOp::Add => prev_val + curr_val,
                    ScanOp::Mul => prev_val * curr_val,
                    ScanOp::Max => if prev_val > curr_val { prev_val } else { curr_val },
                    ScanOp::Min => if prev_val < curr_val { prev_val } else { curr_val },
                    _ => curr_val,
                };
            }
        }
        
        sync_cube();
        
        // Step 3: Get chunk totals and do parallel scan on them
        let mut shared_totals = SharedMemory::<F>::new(256);
        
        // Each thread stores its chunk's last value
        let last_idx = thread_id * chunk_size + chunk_size - 1;
        if last_idx < scan_dim_size {
            let last_element_idx = base_offset + last_idx * stride;
            shared_totals[thread_id] = output[last_element_idx];
        } else {
            shared_totals[thread_id] = match operation {
                ScanOp::Add => F::new(0.0),
                ScanOp::Mul => F::new(1.0),
                _ => F::new(0.0),
            };
        }
        
        sync_cube();
        
        // Parallel scan on chunk totals - Hillis-Steele
        let mut step = 1u32;
        while step < cube_size {
            if thread_id >= step {
                let prev_val = shared_totals[thread_id - step];
                let curr_val = shared_totals[thread_id];
                
                shared_totals[thread_id] = match operation {
                    ScanOp::Add => prev_val + curr_val,
                    ScanOp::Mul => prev_val * curr_val,
                    ScanOp::Max => if prev_val > curr_val { prev_val } else { curr_val },
                    ScanOp::Min => if prev_val < curr_val { prev_val } else { curr_val },
                    _ => curr_val,
                };
            }
            step *= 2;
            sync_cube();
        }
        
        // Step 4: Add prefix to all elements in later chunks
        if thread_id > 0 {
            let prefix = shared_totals[thread_id - 1];
            
            for i in 0..chunk_size {
                let idx = thread_id * chunk_size + i;
                if idx < scan_dim_size {
                    let element_idx = base_offset + idx * stride;
                    let curr_val = output[element_idx];
                    
                    output[element_idx] = match operation {
                        ScanOp::Add => prefix + curr_val,
                        ScanOp::Mul => prefix * curr_val,
                        ScanOp::Max => if prefix > curr_val { prefix } else { curr_val },
                        ScanOp::Min => if prefix < curr_val { prefix } else { curr_val },
                        _ => curr_val,
                    };
                }
            }
        }
    }
}

/// GPU scan operation dispatch  
/// Always uses GPU parallel scan - no automatic switching
pub fn gpu_scan_parallel<R: CubeRuntime, E: CubeElement>(
    tensor: CubeTensor<R>,
    config: ScanConfig,
) -> CubeTensor<R> {
    // Use the simplified parallel kernel
    let output = empty_device::<R, E>(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape.clone(),
    );
    
    let scan_dim_size = tensor.shape.dims[config.dim];
    let total_elements = tensor.shape.num_elements();
    
    // Handle empty tensors
    if total_elements == 0 || scan_dim_size == 0 {
        return output;
    }
    
    let num_scan_lines = total_elements / scan_dim_size;
    
    let cube_dim = CubeDim::new(256u32, 1, 1);
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
        burn_tensor::DType::F16 => {
            let input_f16 = tensor.as_tensor_arg::<half::f16>(1);
            let output_f16 = output.as_tensor_arg::<half::f16>(1);
            
            unsafe {
                scan_parallel_dim_kernel::launch_unchecked::<half::f16, R>(
                    &tensor.client,
                    cube_count,
                    cube_dim,
                    input_f16,
                    output_f16,
                    config.dim as u32,
                    config.op,
                );
            }
        }
        _ => panic!("Unsupported data type for GPU scan"),
    }
    
    output
}

