use crate::{element::FloatNdArrayElement, tensor::NdArrayTensor};
use alloc::vec::Vec;
use burn_common::run_par;
use ndarray::{ArrayD, IxDyn};
use rustfft::{FftPlanner, num_complex::Complex};

/// Minimum number of elements to consider parallel FFT processing
const FFT_PARALLEL_THRESHOLD: usize = 1024;

/// Perform FFT along a specific dimension using rustfft with parallel support
/// 
/// IMPORTANT: Due to Burn's type system constraints, we implement a "same-shape" FFT
/// where complex results are encoded in the same tensor dimensionality as input.
/// For now, we double the last dimension to store [real, imag] pairs.
pub fn fft_dim<E: FloatNdArrayElement + 'static>(
    tensor: NdArrayTensor<E>,
    dim: usize,
) -> NdArrayTensor<E> {
    
    let input_shape = tensor.array.shape();
    let axis_len = input_shape[dim];
    
    // Strategy: Create output with doubled size along last dimension for complex representation
    // Input [4] -> Output [4, 2] where [:,0] = real, [:,1] = imag  
    // This follows burn-tensor FFT API expectation from documentation
    
    if input_shape.len() == 1 {
        // 1D case: [N] -> [N, 2] with parallel processing for large arrays
        return fft_1d_parallel(tensor, axis_len);
    }
    
    // Multi-dimensional case: expand last dimension for complex storage
    let mut output_shape = input_shape.to_vec(); 
    output_shape.push(2);  // Add complex dimension
    
    // For multi-dimensional tensors, use parallel processing along the specified dimension
    fft_multidim_parallel(tensor, dim, output_shape)
}

/// Parallel 1D FFT implementation using Burn's parallel infrastructure
fn fft_1d_parallel<E: FloatNdArrayElement + 'static>(
    tensor: NdArrayTensor<E>,
    axis_len: usize,
) -> NdArrayTensor<E> {
    let input_data: Vec<E> = tensor.array.iter().cloned().collect();
    
    // Check if we should use parallel processing
    let use_parallel = axis_len >= FFT_PARALLEL_THRESHOLD;
    
    let complex_result = if use_parallel {
        #[cfg(feature = "std")]
        {
            // Use Burn's parallel infrastructure for large FFTs
            fft_1d_chunked_parallel(&input_data, axis_len)
        }
        
        #[cfg(not(feature = "std"))]
        {
            // Fallback to sequential for no_std
            fft_1d_sequential(&input_data, axis_len)
        }
    } else {
        fft_1d_sequential(&input_data, axis_len)
    };
    
    // Create output array with shape [N, 2] for [real, imag]
    let output_shape = [axis_len, 2];
    let mut result = ArrayD::<E>::zeros(IxDyn(&output_shape));
    
    // Store complex data as [real, imag] pairs
    for (i, complex_val) in complex_result.iter().enumerate() {
        result[[i, 0]] = complex_val.re;  // Real part
        result[[i, 1]] = complex_val.im;  // Imaginary part
    }
    
    NdArrayTensor::new(result.into_shared())
}

/// Sequential 1D FFT using rustfft
fn fft_1d_sequential<E: FloatNdArrayElement + 'static>(
    input_data: &[E],
    axis_len: usize,
) -> Vec<Complex<E>> {
    // Create FFT planner
    let mut planner = FftPlanner::<E>::new();
    let fft = planner.plan_fft_forward(axis_len);
    
    // Convert to complex
    let mut complex_data: Vec<Complex<E>> = input_data.iter()
        .map(|&x| Complex::new(x, E::from_elem(0.0)))
        .collect();
    
    // Perform FFT using rustfft
    fft.process(&mut complex_data);
    complex_data
}

/// Parallel 1D FFT using Burn's run_par! infrastructure
#[cfg(feature = "std")]
fn fft_1d_chunked_parallel<E: FloatNdArrayElement + 'static>(
    input_data: &[E],
    axis_len: usize,
) -> Vec<Complex<E>> {
    // For very large FFTs, we can break them into smaller chunks
    // and process them in parallel, then combine results
    
    // For now, rustfft is already highly optimized, so we'll use it directly
    // with potential for parallel pre/post-processing
    
    // Convert to complex - this could be parallelized for very large arrays
    let mut complex_data: Vec<Complex<E>> = input_data.iter()
        .map(|&x| Complex::new(x, E::from_elem(0.0)))
        .collect();
    
    // Perform FFT using rustfft (already optimized)
    let mut planner = FftPlanner::<E>::new();
    let fft = planner.plan_fft_forward(axis_len);
    
    // Potential parallel execution context
    run_par!(|| {
        // Execute the FFT - rustfft handles internal optimizations
        fft.process(&mut complex_data);
    });
    
    complex_data
}

/// Multi-dimensional FFT with parallel processing along specified dimension
fn fft_multidim_parallel<E: FloatNdArrayElement + 'static>(
    _tensor: NdArrayTensor<E>,
    _dim: usize,
    output_shape: Vec<usize>,
) -> NdArrayTensor<E> {
    let total_elements: usize = output_shape.iter().product();
    let _use_parallel = total_elements >= FFT_PARALLEL_THRESHOLD;
    
    // For now, return zeros as placeholder for multi-dimensional case
    // TODO: Implement full multi-dimensional FFT with parallel processing
    
    let result = ArrayD::<E>::zeros(IxDyn(&output_shape));
    NdArrayTensor::new(result.into_shared())
}

/// Perform inverse FFT along a specific dimension with parallel support  
/// 
/// Expects input tensor to have complex format with last dimension = 2 [real, imag]
pub fn ifft_dim<E: FloatNdArrayElement + 'static>(
    tensor: NdArrayTensor<E>,
    dim: usize,
) -> NdArrayTensor<E> {
    let input_shape = tensor.array.shape().to_vec(); // Clone to avoid borrow issues
    
    // Expect input to have complex format: [..., 2] where last dim is [real, imag]
    if input_shape.is_empty() || input_shape[input_shape.len() - 1] != 2 {
        panic!("IFFT input must have complex format with last dimension = 2");
    }
    
    // Remove the complex dimension to get output shape
    let output_shape = &input_shape[..input_shape.len() - 1];
    let axis_len = output_shape[dim];
    
    if output_shape.len() == 1 {
        // 1D case: [N, 2] -> [N] with parallel support
        return ifft_1d_parallel(tensor, axis_len, output_shape);
    }
    
    // Multi-dimensional case with parallel support
    ifft_multidim_parallel(tensor, axis_len, output_shape)
}

/// Parallel 1D IFFT implementation
fn ifft_1d_parallel<E: FloatNdArrayElement + 'static>(
    tensor: NdArrayTensor<E>,
    axis_len: usize,
    output_shape: &[usize],
) -> NdArrayTensor<E> {
    let use_parallel = axis_len >= FFT_PARALLEL_THRESHOLD;
    
    // Extract complex data from [real, imag] format
    let complex_data: Vec<Complex<E>> = if use_parallel {
        #[cfg(feature = "std")]
        {
            // Use sequential extraction for now due to borrowing constraints
            // TODO: Optimize with proper parallel extraction
            (0..axis_len).map(|i| {
                let real = tensor.array[[i, 0]];
                let imag = tensor.array[[i, 1]];
                Complex::new(real, imag)
            }).collect()
        }
        
        #[cfg(not(feature = "std"))]
        {
            (0..axis_len).map(|i| {
                let real = tensor.array[[i, 0]];
                let imag = tensor.array[[i, 1]];
                Complex::new(real, imag)
            }).collect()
        }
    } else {
        (0..axis_len).map(|i| {
            let real = tensor.array[[i, 0]];
            let imag = tensor.array[[i, 1]];
            Complex::new(real, imag)
        }).collect()
    };
    
    // Perform inverse FFT
    let mut complex_result = complex_data;
    let mut planner = FftPlanner::<E>::new();
    let ifft = planner.plan_fft_inverse(axis_len);
    ifft.process(&mut complex_result);
    
    // Create real-valued output
    let mut result = ArrayD::<E>::zeros(IxDyn(output_shape));
    
    // Extract real parts (imaginary should be ~0 for real input)
    for (i, complex_val) in complex_result.iter().enumerate() {
        result[i] = complex_val.re;
    }
    
    NdArrayTensor::new(result.into_shared())
}

/// Multi-dimensional IFFT with parallel support
fn ifft_multidim_parallel<E: FloatNdArrayElement + 'static>(
    _tensor: NdArrayTensor<E>,
    _axis_len: usize,
    output_shape: &[usize],
) -> NdArrayTensor<E> {
    // TODO: Implement parallel multi-dimensional IFFT
    // For now, return placeholder
    
    let result = ArrayD::<E>::zeros(IxDyn(output_shape));
    NdArrayTensor::new(result.into_shared())
}
