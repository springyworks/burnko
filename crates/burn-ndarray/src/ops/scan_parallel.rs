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
    
    /// Large-scale performance test comparing different tensor sizes
    #[test]
    fn test_large_scale_parallel_performance() {
        let device = Default::default();
        let core_count = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        
        println!("🧵 Running large-scale parallel scan performance test on {} cores", core_count);
        
        let test_sizes = vec![
            1_000,     // Should be sequential (below threshold)
            10_000,    // Should trigger parallel
            100_000,   // Definitely parallel
            1_000_000, // Large parallel workload
        ];
        
        for &size in &test_sizes {
            println!("\n📊 Testing size: {} elements", size);
            
            // Create test tensor with positive uniform random values (0.1 to 1.0) for monotonic cumsum
            let tensor: Tensor<TestBackend, 1> = Tensor::random(
                [size], Distribution::Uniform(0.1, 1.0), &device
            );
            
            // Test cumsum performance
            let start = Instant::now();
            let result = tensor.clone().cumsum(0);
            let duration = start.elapsed();
            
            let throughput = size as f64 / duration.as_secs_f64();
            println!("   ⚡ Cumsum: {:?} ({:.2}M elements/sec)", duration, throughput / 1_000_000.0);
            
            // Verify correctness for cumsum with positive values - should be monotonic
            let values: Vec<f32> = result.to_data().to_vec().unwrap();
            // Since all values are positive (0.1 to 1.0), cumsum should be strictly monotonic
            for i in 1..values.len() {
                assert!(values[i] >= values[i-1], 
                        "Cumsum monotonic property violated at index {}: {} < {}", i, values[i], values[i-1]);
            }
            
            // Test cumprod performance (with smaller values to avoid overflow)
            let small_tensor: Tensor<TestBackend, 1> = Tensor::random(
                [size], Distribution::Uniform(1.0, 1.001), &device
            );
            
            let start = Instant::now();
            let _prod_result = small_tensor.cumprod(0);
            let duration = start.elapsed();
            
            let throughput = size as f64 / duration.as_secs_f64();
            println!("   ⚡ Cumprod: {:?} ({:.2}M elements/sec)", duration, throughput / 1_000_000.0);
        }
        
        println!("\n🎯 Large-scale parallel performance test completed successfully!");
    }
}
