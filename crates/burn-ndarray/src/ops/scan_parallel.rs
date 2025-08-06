//! Scan operations for ndarray backend
//! 
//! Cumulative sum and product operations using ndarray's built-in implementations.

use crate::{element::NdArrayElement, tensor::NdArrayTensor};
use ndarray::Axis;

/// Parallel cumulative sum along the specified dimension
pub(crate) fn cumsum_dim_parallel<E: NdArrayElement>(
    tensor: NdArrayTensor<E>,
    dim: usize,
) -> NdArrayTensor<E> {
    let axis = Axis(dim);
    let mut array = tensor.array.into_owned();
    
    // Use ndarray's built-in accumulate_axis_inplace which is correct
    array.accumulate_axis_inplace(axis, |&prev, curr| *curr = *curr + prev);
    
    NdArrayTensor::new(array.into_shared())
}

/// Parallel cumulative product along the specified dimension
pub(crate) fn cumprod_dim_parallel<E: NdArrayElement>(
    tensor: NdArrayTensor<E>,
    dim: usize,
) -> NdArrayTensor<E> {
    let axis = Axis(dim);
    let mut array = tensor.array.into_owned();
    
    // Use ndarray's built-in accumulate_axis_inplace which is correct
    array.accumulate_axis_inplace(axis, |&prev, curr| *curr = *curr * prev);
    
    NdArrayTensor::new(array.into_shared())
}

#[cfg(test)]
mod tests {
    use crate::NdArray;
    use burn_tensor::{Tensor, TensorData, Shape};
    
    type TestBackend = NdArray<f32>;
    
    #[test]
    fn test_cumsum_basic() {
        let device = Default::default();
        
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor: Tensor<TestBackend, 1> = Tensor::from_data(
            TensorData::new(data, Shape::new([5])), &device
        );
        
        let result = tensor.cumsum(0);
        let output: Vec<f32> = result.to_data().to_vec().unwrap();
        let expected = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        
        assert_eq!(output, expected);
    }
}
