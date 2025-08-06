//! GPU FFT operations using CubeCL compute shaders
//! 
//! This module implements Fast Fourier Transform operations for GPU execution
//! following Burn's established CubeCL patterns and the gpu-fft library approach.

#![allow(missing_docs)] // Allow missing docs for cube macro-generated code

use crate::{CubeRuntime, element::CubeElement, ops::numeric::empty_device, tensor::CubeTensor};
use burn_tensor::Shape;
use cubecl::{calculate_cube_count_elemwise, prelude::*};
use std::f32::consts::PI;

/// Basic FFT kernel using direct DFT computation
/// Each thread computes one output frequency bin
#[cube(launch_unchecked)]
pub fn fft_basic_kernel<F: Float>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] fft_size: u32,
    #[comptime] is_inverse: bool,
) {
    let idx = ABSOLUTE_POS;
    
    if idx >= fft_size {
        terminate!();
    }
    
    let mut real = Line::<F>::new(F::new(0.0));
    let mut imag = Line::<F>::new(F::new(0.0));
    
    // Choose sign based on forward/inverse FFT
    let sign = if is_inverse { F::new(2.0) } else { F::new(-2.0) };
    let angle_increment = sign * F::new(PI) / F::cast_from(fft_size);
    
    // Direct DFT computation
    for k in 0..fft_size {
        let input_val = input[k];
        let angle = angle_increment * F::cast_from(k) * F::cast_from(idx);
        let cos_angle = F::cos(angle);
        let sin_angle = F::sin(angle);
        
        // Complex multiplication: input_val * (cos_angle + i*sin_angle)
        // Since input is real, imag part = input_val * sin_angle
        real += input_val * Line::new(cos_angle);
        imag += input_val * Line::new(sin_angle);
    }
    
    // Normalization for inverse FFT
    if is_inverse {
        let norm = F::new(1.0) / F::cast_from(fft_size);
        real *= Line::new(norm);
        imag *= Line::new(norm);
    }
    
    // Store in interleaved format: [real0, imag0, real1, imag1, ...]
    output[idx * 2] = real;
    output[idx * 2 + 1] = imag;
}

/// Optimized FFT kernel for larger sizes (placeholder for future Cooley-Tukey implementation)
#[cube(launch_unchecked)]
pub fn fft_optimized_kernel<F: Float>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] fft_size: u32,
    #[comptime] is_inverse: bool,
) {
    // For now, use the same basic implementation
    // TODO: Implement proper radix-2 Cooley-Tukey FFT with shared memory
    let idx = ABSOLUTE_POS;
    
    if idx >= fft_size {
        terminate!();
    }
    
    let mut real = Line::<F>::new(F::new(0.0));
    let mut imag = Line::<F>::new(F::new(0.0));
    
    let sign = if is_inverse { F::new(2.0) } else { F::new(-2.0) };
    let angle_increment = sign * F::new(PI) / F::cast_from(fft_size);
    
    for k in 0..fft_size {
        let input_val = input[k];
        let angle = angle_increment * F::cast_from(k) * F::cast_from(idx);
        let cos_angle = F::cos(angle);
        let sin_angle = F::sin(angle);
        
        real += input_val * Line::new(cos_angle);
        imag += input_val * Line::new(sin_angle);
    }
    
    if is_inverse {
        let norm = F::new(1.0) / F::cast_from(fft_size);
        real *= Line::new(norm);
        imag *= Line::new(norm);
    }
    
    output[idx * 2] = real;
    output[idx * 2 + 1] = imag;
}

/// GPU FFT function following Burn's CubeTensor patterns
pub fn gpu_fft<R: CubeRuntime, E: CubeElement>(
    tensor: CubeTensor<R>,
    fft_dim: usize,
) -> CubeTensor<R> {
    let fft_size = tensor.shape.dims[fft_dim];
    
    // Create output tensor with complex shape [..., N, 2] for interleaved complex
    let mut output_shape_dims = tensor.shape.dims.clone();
    output_shape_dims.push(2); // Add dimension for real/imaginary interleaved
    let output_shape = Shape::from(output_shape_dims);
    
    let output = empty_device::<R, E>(
        tensor.client.clone(),
        tensor.device.clone(),
        output_shape,
    );
    
    // Calculate launch parameters
    let cube_dim = CubeDim::new(256, 1, 1); // Use 256 threads per workgroup
    let cube_count = calculate_cube_count_elemwise(fft_size, cube_dim);
    
    match E::dtype() {
        burn_tensor::DType::F32 => {
            let input_arg = tensor.as_tensor_arg::<f32>(1);
            let output_arg = output.as_tensor_arg::<f32>(1);
            
            // Choose kernel based on FFT size
            if fft_size <= 1024 {
                unsafe {
                    fft_basic_kernel::launch_unchecked::<f32, R>(
                        &tensor.client,
                        cube_count,
                        cube_dim,
                        input_arg,
                        output_arg,
                        fft_size as u32,
                        false, // forward FFT
                    );
                }
            } else {
                unsafe {
                    fft_optimized_kernel::launch_unchecked::<f32, R>(
                        &tensor.client,
                        cube_count,
                        cube_dim,
                        input_arg,
                        output_arg,
                        fft_size as u32,
                        false,
                    );
                }
            }
        }
        burn_tensor::DType::F64 => {
            let input_arg = tensor.as_tensor_arg::<f64>(1);
            let output_arg = output.as_tensor_arg::<f64>(1);
            
            if fft_size <= 1024 {
                unsafe {
                    fft_basic_kernel::launch_unchecked::<f64, R>(
                        &tensor.client,
                        cube_count,
                        cube_dim,
                        input_arg,
                        output_arg,
                        fft_size as u32,
                        false,
                    );
                }
            } else {
                unsafe {
                    fft_optimized_kernel::launch_unchecked::<f64, R>(
                        &tensor.client,
                        cube_count,
                        cube_dim,
                        input_arg,
                        output_arg,
                        fft_size as u32,
                        false,
                    );
                }
            }
        }
        _ => panic!("Unsupported data type for GPU FFT: {:?}", E::dtype()),
    }
    
    output
}

/// GPU inverse FFT function
pub fn gpu_ifft<R: CubeRuntime, E: CubeElement>(
    tensor: CubeTensor<R>,
    fft_dim: usize,
) -> CubeTensor<R> {
    // Input tensor should have shape [..., N, 2] for complex data
    // Output tensor will have shape [..., N] for real data  
    let fft_size = tensor.shape.dims[fft_dim];
    
    // TODO: Implement proper complex-to-real IFFT
    // For now, this is a placeholder that assumes real input
    // A proper implementation would:
    // 1. Read complex input from interleaved format
    // 2. Perform inverse FFT 
    // 3. Extract only real parts for output
    
    // Create output tensor with real shape (remove complex dimension)
    let mut output_shape_dims = tensor.shape.dims.clone();
    if output_shape_dims.len() > 1 && output_shape_dims[output_shape_dims.len() - 1] == 2 {
        output_shape_dims.pop(); // Remove complex dimension if present
    }
    let output_shape = Shape::from(output_shape_dims);
    
    let output = empty_device::<R, E>(
        tensor.client.clone(),
        tensor.device.clone(),
        output_shape,
    );
    
    let cube_dim = CubeDim::new(256, 1, 1);
    let cube_count = calculate_cube_count_elemwise(fft_size, cube_dim);
    
    // TODO: Replace with proper IFFT kernel that handles complex input
    match E::dtype() {
        burn_tensor::DType::F32 => {
            let input_arg = tensor.as_tensor_arg::<f32>(1);
            let output_arg = output.as_tensor_arg::<f32>(1);
            
            unsafe {
                fft_basic_kernel::launch_unchecked::<f32, R>(
                    &tensor.client,
                    cube_count,
                    cube_dim,
                    input_arg,
                    output_arg,
                    fft_size as u32,
                    true, // inverse FFT
                );
            }
        }
        burn_tensor::DType::F64 => {
            let input_arg = tensor.as_tensor_arg::<f64>(1);
            let output_arg = output.as_tensor_arg::<f64>(1);
            
            unsafe {
                fft_basic_kernel::launch_unchecked::<f64, R>(
                    &tensor.client,
                    cube_count,
                    cube_dim,
                    input_arg,
                    output_arg,
                    fft_size as u32,
                    true,
                );
            }
        }
        _ => panic!("Unsupported data type for GPU IFFT: {:?}", E::dtype()),
    }
    
    output
}
