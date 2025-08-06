// Debug test to understand the axis iteration issue

use burn_ndarray::{NdArray, NdArrayDevice};
use burn_tensor::{Tensor, TensorData, Shape};

type TestBackend = NdArray<f32>;

#[test]
fn debug_axis_iteration() {
    let device = NdArrayDevice::default();
    
    println!("=== Debugging Axis Iteration Issue ===");
    
    // Create a simple 2x3 matrix to understand the axis issue
    let data = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new([2, 3]));
    let tensor = Tensor::<TestBackend, 2>::from_data(data.convert::<f32>(), &device);
    
    println!("Input matrix (2x3):");
    println!("[[1, 2, 3],");
    println!(" [4, 5, 6]]");
    
    // Test cumsum along axis 0 (should sum down columns)
    println!("\nTesting cumsum along axis 0 (columns):");
    println!("Expected: [[1, 2, 3], [5, 7, 9]]");
    let result_axis0 = tensor.clone().cumsum(0);
    let values_axis0 = result_axis0.to_data().to_vec::<f32>().unwrap();
    println!("Actual: {:?}", values_axis0);
    
    // Test cumsum along axis 1 (should sum across rows) 
    println!("\nTesting cumsum along axis 1 (rows):");
    println!("Expected: [[1, 3, 6], [4, 9, 15]]");
    let result_axis1 = tensor.cumsum(1);
    let values_axis1 = result_axis1.to_data().to_vec::<f32>().unwrap();
    println!("Actual: {:?}", values_axis1);
    
    // Compare with what ndarray's accumulate_axis_inplace would give us
    use ndarray::{Array2, Axis};
    let ndarray_test = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    println!("\nDirect ndarray accumulate_axis_inplace test:");
    let mut ndarray_axis0 = ndarray_test.clone();
    ndarray_axis0.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr = *curr + prev);
    println!("ndarray axis 0: {:?}", ndarray_axis0.as_slice().unwrap());
    
    let mut ndarray_axis1 = ndarray_test.clone();
    ndarray_axis1.accumulate_axis_inplace(Axis(1), |&prev, curr| *curr = *curr + prev);
    println!("ndarray axis 1: {:?}", ndarray_axis1.as_slice().unwrap());
}
