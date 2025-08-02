//! Parallel scan operations for ndarray backend using Rayon
//! 
//! This module provides parallel implementations of cumulative operations
//! using Rayon's work-stealing scheduler.
//! 
//! Features:
//! - Threshold-based selection between parallel and sequential execution
//! - Divide-and-conquer parallel prefix computation
//! - Rayon-based chunk processing
//! - Support for cumsum and cumprod operations

use crate::{element::NdArrayElement, tensor::NdArrayTensor};
use ndarray::{Axis, Dimension, RemoveAxis, ArrayBase, DataMut};
use num_traits::{Zero, One};

#[cfg(feature = "std")]
use burn_common::rayon::prelude::*;

/// Minimum number of elements to consider parallel processing
/// Threshold for parallel execution
/// Below this size, sequential processing is used for better performance
const PARALLEL_THRESHOLD: usize = 1000;

/// Minimum elements per thread for efficient parallelization
const MIN_ELEMENTS_PER_THREAD: usize = 1000;

/// Parallel cumulative sum along the specified dimension using Rayon
/// 
/// Uses divide-and-conquer approach for parallel prefix computation:
/// 1. Chunks array into parallel segments  
/// 2. Computes prefix sums within each chunk
/// 3. Propagates prefix values between chunks
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

/// Parallel cumulative product implementation using divide-and-conquer strategy  
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

/// In-place parallel cumulative sum using aggressive multi-core approach
#[cfg(feature = "std")]
fn parallel_cumsum_inplace<E, S, D>(array: &mut ArrayBase<S, D>, axis: Axis)
where
    E: NdArrayElement + Zero + Send + Sync,
    S: DataMut<Elem = E>,
    D: Dimension + RemoveAxis,
{
    let num_threads = burn_common::rayon::current_num_threads();
    let axis_len = array.shape()[axis.index()];
    
    if axis_len <= 1 {
        return;
    }
    
    // For MAXIMUM parallelization: process each lane/row in parallel
    // AND parallelize within large lanes using divide-and-conquer
    array.axis_iter_mut(axis)
        .into_par_iter()
        .for_each(|mut lane| {
            let lane_len = lane.len();
            
            if lane_len > MIN_ELEMENTS_PER_THREAD * num_threads {
                // Large lane: use aggressive divide-and-conquer parallel scan
                parallel_scan_large_lane(lane.as_slice_mut().unwrap());
            } else {
                // Small lane: sequential scan
                let mut acc = E::zero();
                for elem in lane.iter_mut() {
                    acc = acc + *elem;
                    *elem = acc;
                }
            }
        });
}

/// Aggressive parallel scan for large data arrays using divide-and-conquer
/// This will saturate ALL available CPU cores
#[cfg(feature = "std")]
fn parallel_scan_large_lane<E>(data: &mut [E])
where
    E: NdArrayElement + Zero + Send + Sync,
{
    let len = data.len();
    let num_threads = burn_common::rayon::current_num_threads();
    
    if len < MIN_ELEMENTS_PER_THREAD * 2 {
        // Too small for parallel processing
        let mut acc = E::zero();
        for elem in data.iter_mut() {
            acc = acc + *elem;
            *elem = acc;
        }
        return;
    }
    
    // Phase 1: Divide into chunks and process in parallel
    let chunk_size = (len + num_threads - 1) / num_threads;
    let mut chunks: Vec<_> = data.chunks_mut(chunk_size).collect();
    
    // Parallel phase: compute prefix sum within each chunk and collect totals
    let chunk_totals: Vec<E> = chunks
        .par_iter_mut()
        .map(|chunk| {
            let mut acc = E::zero();
            for elem in chunk.iter_mut() {
                acc = acc + *elem;
                *elem = acc;
            }
            acc  // Return the total for this chunk
        })
        .collect();
    
    // Phase 2: Compute prefix sum of chunk totals (sequential - small)
    let mut chunk_prefixes = vec![E::zero(); chunk_totals.len()];
    let mut running_total = E::zero();
    for (i, &total) in chunk_totals.iter().enumerate() {
        chunk_prefixes[i] = running_total;
        running_total = running_total + total;
    }
    
    // Phase 3: Add chunk prefixes to all elements in parallel
    chunks
        .into_par_iter()
        .zip(chunk_prefixes.par_iter())
        .for_each(|(chunk, &prefix)| {
            if !prefix.is_zero() {
                for elem in chunk {
                    *elem = *elem + prefix;
                }
            }
        });
}

/// In-place parallel cumulative product using aggressive multi-core approach
#[cfg(feature = "std")]
fn parallel_cumprod_inplace<E, S, D>(array: &mut ArrayBase<S, D>, axis: Axis)
where
    E: NdArrayElement + One + Send + Sync,
    S: DataMut<Elem = E>,
    D: Dimension + RemoveAxis,
{
    let num_threads = burn_common::rayon::current_num_threads();
    let axis_len = array.shape()[axis.index()];
    
    if axis_len <= 1 {
        return;
    }
    
    // For MAXIMUM parallelization: process each lane/row in parallel
    // AND parallelize within large lanes using divide-and-conquer
    array.axis_iter_mut(axis)
        .into_par_iter()
        .for_each(|mut lane| {
            let lane_len = lane.len();
            
            if lane_len > MIN_ELEMENTS_PER_THREAD * num_threads {
                // Large lane: use aggressive divide-and-conquer parallel scan
                parallel_scan_large_lane_prod(lane.as_slice_mut().unwrap());
            } else {
                // Small lane: sequential scan
                let mut acc = E::one();
                for elem in lane.iter_mut() {
                    acc = acc * *elem;
                    *elem = acc;
                }
            }
        });
}

/// Aggressive parallel scan for large data arrays using divide-and-conquer (product version)
#[cfg(feature = "std")]
fn parallel_scan_large_lane_prod<E>(data: &mut [E])
where
    E: NdArrayElement + One + Send + Sync,
{
    let len = data.len();
    let num_threads = burn_common::rayon::current_num_threads();
    
    if len < MIN_ELEMENTS_PER_THREAD * 2 {
        // Too small for parallel processing
        let mut acc = E::one();
        for elem in data.iter_mut() {
            acc = acc * *elem;
            *elem = acc;
        }
        return;
    }
    
    // Phase 1: Divide into chunks and process in parallel
    let chunk_size = (len + num_threads - 1) / num_threads;
    let mut chunks: Vec<_> = data.chunks_mut(chunk_size).collect();
    
    // Parallel phase: compute prefix product within each chunk and collect totals
    let chunk_totals: Vec<E> = chunks
        .par_iter_mut()
        .map(|chunk| {
            let mut acc = E::one();
            for elem in chunk.iter_mut() {
                acc = acc * *elem;
                *elem = acc;
            }
            acc  // Return the total for this chunk
        })
        .collect();
    
    // Phase 2: Compute prefix product of chunk totals (sequential - small)
    let mut chunk_prefixes = vec![E::one(); chunk_totals.len()];
    let mut running_total = E::one();
    for (i, &total) in chunk_totals.iter().enumerate() {
        chunk_prefixes[i] = running_total;
        running_total = running_total * total;
    }
    
    // Phase 3: Multiply chunk prefixes with all elements in parallel
    chunks
        .into_par_iter()
        .zip(chunk_prefixes.par_iter())
        .for_each(|(chunk, &prefix)| {
            if !prefix.is_one() {
                for elem in chunk {
                    *elem = *elem * prefix;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_parallel_cumsum_simple() {
        let data = Array2::from_shape_vec((2, 4), vec![1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let tensor = NdArrayTensor::new(data.into_dyn().into_shared());
        
        let result = cumsum_dim_parallel(tensor, 1);
        
        // Expected: [[1, 3, 6, 10], [5, 11, 18, 26]]
        assert_eq!(result.array[[0, 0]], 1);
        assert_eq!(result.array[[0, 1]], 3);
        assert_eq!(result.array[[0, 2]], 6);
        assert_eq!(result.array[[0, 3]], 10);
        assert_eq!(result.array[[1, 0]], 5);
        assert_eq!(result.array[[1, 1]], 11);
        assert_eq!(result.array[[1, 2]], 18);
        assert_eq!(result.array[[1, 3]], 26);
    }
    
    #[test]
    fn test_parallel_cumprod_simple() {
        let data = Array2::from_shape_vec((2, 4), vec![1, 2, 3, 4, 1, 2, 3, 4]).unwrap();
        let tensor = NdArrayTensor::new(data.into_dyn().into_shared());
        
        let result = cumprod_dim_parallel(tensor, 1);
        
        // Expected: [[1, 2, 6, 24], [1, 2, 6, 24]]
        assert_eq!(result.array[[0, 0]], 1);
        assert_eq!(result.array[[0, 1]], 2);
        assert_eq!(result.array[[0, 2]], 6);
        assert_eq!(result.array[[0, 3]], 24);
        assert_eq!(result.array[[1, 0]], 1);
        assert_eq!(result.array[[1, 1]], 2);
        assert_eq!(result.array[[1, 2]], 6);
        assert_eq!(result.array[[1, 3]], 24);
    }
    
    #[test]
    fn test_parallel_threshold() {
        // Small array should use sequential path
        let data = Array2::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let tensor = NdArrayTensor::new(data.into_dyn().into_shared());
        
        let result = cumsum_dim_parallel(tensor, 1);
        
        // Expected: [[1, 3, 6], [4, 9, 15]]
        assert_eq!(result.array[[0, 0]], 1);
        assert_eq!(result.array[[0, 1]], 3);
        assert_eq!(result.array[[0, 2]], 6);
        assert_eq!(result.array[[1, 0]], 4);
        assert_eq!(result.array[[1, 1]], 9);
        assert_eq!(result.array[[1, 2]], 15);
    }
}