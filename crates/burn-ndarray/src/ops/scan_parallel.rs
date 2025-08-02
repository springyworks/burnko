//! Parallel scan operations for ndarray backend using Burn's parallel infrastructure
//! 
//! This module provides parallel implementations of cumulative operations
//! following Burn's established patterns using run_par! and iter_par! macros.
//! 
//! Features:
//! - Threshold-based selection between parallel and sequential execution
//! - Integration with Burn's standard parallel infrastructure
//! - Multi-core utilization for large tensor operations
//! - Support for cumsum and cumprod operations

use crate::{element::NdArrayElement, tensor::NdArrayTensor};
use ndarray::{Axis, Dimension, RemoveAxis, ArrayBase, DataMut};
use num_traits::{Zero, One};
use burn_common::{run_par, iter_par};

/// Minimum number of elements to consider parallel processing
/// Threshold for parallel execution
/// Below this size, sequential processing is used for better performance
const PARALLEL_THRESHOLD: usize = 1000;

/// Minimum elements per thread for efficient parallelization
const MIN_ELEMENTS_PER_THREAD: usize = 1000;

/// Parallel cumulative sum along the specified dimension
/// 
/// Uses Burn's parallel infrastructure for multi-core processing:
/// 1. Processes multiple scan lines in parallel using iter_par!
/// 2. Uses sequential processing within each line for correctness
/// 3. Falls back to sequential processing for small arrays
pub(crate) fn cumsum_dim_parallel<E: NdArrayElement>(
    tensor: NdArrayTensor<E>,
    dim: usize,
) -> NdArrayTensor<E> {
    let axis = Axis(dim);
    let mut array = tensor.array.into_owned();
    
    // Check if parallel processing is beneficial
    let axis_len = array.shape()[dim];
    if axis_len < PARALLEL_THRESHOLD {
        // Use sequential implementation for small arrays
        array.accumulate_axis_inplace(axis, |&prev, curr| *curr = *curr + prev);
        return NdArrayTensor::new(array.into_shared());
    }
    
    #[cfg(feature = "std")]
    {
        parallel_cumsum_inplace(&mut array, axis);
    }
    
    #[cfg(not(feature = "std"))]
    {
        array.accumulate_axis_inplace(axis, |&prev, curr| *curr = *curr + prev);
    }
    
    NdArrayTensor::new(array.into_shared())
}

/// Parallel cumulative product along the specified dimension
pub(crate) fn cumprod_dim_parallel<E: NdArrayElement>(
    tensor: NdArrayTensor<E>,
    dim: usize,
) -> NdArrayTensor<E> {
    let axis = Axis(dim);
    let mut array = tensor.array.into_owned();
    
    // Check if parallel processing is beneficial
    let axis_len = array.shape()[dim];
    if axis_len < PARALLEL_THRESHOLD {
        // Use sequential implementation for small arrays
        array.accumulate_axis_inplace(axis, |&prev, curr| *curr = *curr * prev);
        return NdArrayTensor::new(array.into_shared());
    }
    
    #[cfg(feature = "std")]
    {
        parallel_cumprod_inplace(&mut array, axis);
    }
    
    #[cfg(not(feature = "std"))]
    {
        array.accumulate_axis_inplace(axis, |&prev, curr| *curr = *curr * prev);
    }
    
    NdArrayTensor::new(array.into_shared())
}

/// In-place parallel cumulative sum using Burn's parallel infrastructure
/// 
/// This follows the same pattern used in conv.rs and other Burn operations:
/// - run_par! for scoped parallel execution  
/// - iter_par! for parallel iteration over axis
/// 
/// Note: For 1D tensors along axis 0, we cannot parallelize effectively since 
/// each axis iteration gives us a single scalar. In such cases, we fall back to sequential.
#[cfg(feature = "std")]
fn parallel_cumsum_inplace<E, S, D>(array: &mut ArrayBase<S, D>, axis: Axis)
where
    E: NdArrayElement + Zero + Send + Sync,
    S: DataMut<Elem = E> + Send,
    D: Dimension + RemoveAxis,
{
    let axis_len = array.shape()[axis.index()];
    
    if axis_len <= 1 {
        return;
    }

    // For effective parallelization, we need multiple independent lanes.
    // If we're doing cumsum along axis 0 on a 1D array, each "lane" is just a scalar,
    // so parallelization doesn't help and can cause correctness issues.
    let total_elements = array.len();
    let lanes_count = total_elements / axis_len;
    
    if lanes_count < 2 {
        // Fall back to sequential for 1D case or when we don't have enough lanes
        array.accumulate_axis_inplace(axis, |&prev, curr| *curr = *curr + prev);
        return;
    }

    // Use Burn's standard parallel pattern - same as conv.rs
    run_par!(|| {
        iter_par!(array.axis_iter_mut(axis))
            .for_each(|mut lane| {
                let slice = lane.as_slice_mut().unwrap();
                sequential_cumsum_slice(slice);
            });
    });
}

/// In-place parallel cumulative product using Burn's parallel infrastructure
#[cfg(feature = "std")]
fn parallel_cumprod_inplace<E, S, D>(array: &mut ArrayBase<S, D>, axis: Axis)
where
    E: NdArrayElement + One + Send + Sync,
    S: DataMut<Elem = E> + Send,
    D: Dimension + RemoveAxis,
{
    let axis_len = array.shape()[axis.index()];
    
    if axis_len <= 1 {
        return;
    }

    // For effective parallelization, we need multiple independent lanes.
    let total_elements = array.len();
    let lanes_count = total_elements / axis_len;
    
    if lanes_count < 2 {
        // Fall back to sequential for 1D case or when we don't have enough lanes
        array.accumulate_axis_inplace(axis, |&prev, curr| *curr = *curr * prev);
        return;
    }

    // Use Burn's standard parallel pattern
    run_par!(|| {
        iter_par!(array.axis_iter_mut(axis))
            .for_each(|mut lane| {
                let slice = lane.as_slice_mut().unwrap();
                sequential_cumprod_slice(slice);
            });
    });
}

/// Sequential cumsum implementation for slices
fn sequential_cumsum_slice<E: NdArrayElement + Zero>(slice: &mut [E]) {
    for i in 1..slice.len() {
        slice[i] = slice[i] + slice[i - 1];
    }
}

/// Sequential cumprod implementation for slices
fn sequential_cumprod_slice<E: NdArrayElement + One>(slice: &mut [E]) {
    for i in 1..slice.len() {
        slice[i] = slice[i] * slice[i - 1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArray;
    use burn_tensor::{Tensor, TensorData, Shape, Distribution};
    use std::time::Instant;
    
    type TestBackend = NdArray<f32>;
    
    /// Test cumsum with analytical cases where we know the exact expected results
    #[test]
    fn test_analytical_cumsum_cases() {
        let device = Default::default();
        
        // Test 1: All ones - cumsum should be [1, 2, 3, 4, 5, ...]
        let ones_data = vec![1.0; 10];
        let ones: Tensor<TestBackend, 1> = Tensor::from_data(
            TensorData::new(ones_data, Shape::new([10])), &device
        );
        
        let cumsum_ones = ones.cumsum(0);
        let values: Vec<f32> = cumsum_ones.to_data().to_vec().unwrap();
        let expected: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        
        for (i, (&actual, &expected)) in values.iter().zip(expected.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-6, 
                    "Cumsum ones failed at index {}: expected {}, got {}", i, expected, actual);
        }
        println!("✅ Analytical Test 1: All ones cumsum = [1, 2, 3, ..., 10]");
    }
    
    /// Test cumsum with sequential integers - known analytical formula
    #[test]
    fn test_analytical_cumsum_sequential() {
        let device = Default::default();
        
        // Test with sequence [1, 2, 3, 4, 5]
        let seq_data: Vec<f32> = (1..=5).map(|x| x as f32).collect();
        let sequence: Tensor<TestBackend, 1> = Tensor::from_data(
            TensorData::new(seq_data, Shape::new([5])), &device
        );
        
        let cumsum_seq = sequence.cumsum(0);
        let values: Vec<f32> = cumsum_seq.to_data().to_vec().unwrap();
        
        // Expected: cumsum([1,2,3,4,5]) = [1, 3, 6, 10, 15]
        // Formula: cumsum[i] = i*(i+1)/2 for 1-based indexing
        let expected = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        
        for (i, (&actual, &expected)) in values.iter().zip(expected.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-6,
                    "Sequential cumsum failed at index {}: expected {}, got {}", i, expected, actual);
        }
        println!("✅ Analytical Test 2: Sequential [1,2,3,4,5] cumsum = [1,3,6,10,15]");
    }
    
    /// Test multi-dimensional scans with analytical results
    #[test]
    fn test_analytical_multidimensional_scan() {
        let device = Default::default();
        
        // Test 3x3 identity matrix
        let identity_data = vec![
            1.0, 0.0, 0.0,  // [1, 0, 0]
            0.0, 1.0, 0.0,  // [0, 1, 0] 
            0.0, 0.0, 1.0   // [0, 0, 1]
        ];
        let identity: Tensor<TestBackend, 2> = Tensor::from_data(
            TensorData::new(identity_data, Shape::new([3, 3])), &device
        );
        
        // Cumsum along dimension 1 (rows)
        let cumsum_rows = identity.cumsum(1);
        let result_data: Vec<f32> = cumsum_rows.to_data().to_vec().unwrap();
        
        // Expected for cumsum along rows:
        // Row 0: [1,0,0] -> [1, 1, 1]
        // Row 1: [0,1,0] -> [0, 1, 1] 
        // Row 2: [0,0,1] -> [0, 0, 1]
        let expected = vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        
        for (i, (&actual, &expected)) in result_data.iter().zip(expected.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-6,
                    "Identity matrix cumsum failed at index {}: expected {}, got {}", i, expected, actual);
        }
        println!("✅ Analytical Test 3: 3x3 identity matrix cumsum along rows verified");
    }
    
    /// Debug test to understand the correctness issue
    #[test]
    fn test_debug_simple_cumsum() {
        let device = Default::default();
        
        // Test 1: Simple 1D case - [1, 1, 1, 1, 1] should become [1, 2, 3, 4, 5]
        let data = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let tensor: Tensor<TestBackend, 1> = Tensor::from_data(
            TensorData::new(data.clone(), Shape::new([5])), &device
        );
        
        println!("Input: {:?}", data);
        
        let result = tensor.cumsum(0);
        let output: Vec<f32> = result.to_data().to_vec().unwrap();
        
        println!("Output: {:?}", output);
        println!("Expected: [1.0, 2.0, 3.0, 4.0, 5.0]");
        
        // Check each element
        for i in 0..5 {
            let expected = (i + 1) as f32;
            assert_eq!(output[i], expected, 
                      "Mismatch at index {}: got {}, expected {}", i, output[i], expected);
        }
    }
    
    /// Test our sequential implementation directly
    #[test]
    fn test_sequential_cumsum_slice() {
        let mut data = vec![1.0f32, 1.0, 1.0, 1.0, 1.0];
        println!("Before sequential_cumsum_slice: {:?}", data);
        
        sequential_cumsum_slice(&mut data);
        
        println!("After sequential_cumsum_slice: {:?}", data);
        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(data, expected);
    }
}
