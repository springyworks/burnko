use burn_ndarray::NdArray;
use burn_tensor::{Tensor, TensorData, Shape};

#[test]
fn test_tensor_dims_behavior() {
    type TestBackend = NdArray;
    
    // Test 1D tensor
    let data1d = TensorData::from([1.0f32, 2.0, 3.0, 4.0]);
    let tensor1d = Tensor::<TestBackend, 1>::from_data(data1d.clone(), &Default::default());
    println!("1D tensor dims: {:?}", tensor1d.dims());
    println!("1D tensor data shape: {:?}", tensor1d.to_data().shape);

    // Test 2D tensor 
    let data2d = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], Shape::new([4, 2]));
    let tensor2d = Tensor::<TestBackend, 2>::from_data(data2d, &Default::default());
    println!("2D tensor dims: {:?}", tensor2d.dims());
    println!("2D tensor data shape: {:?}", tensor2d.to_data().shape);
    
    // Test manual shape mismatch - what happens if we create wrong type?
    let wrong_data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], Shape::new([4, 2]));
    let wrong_tensor = Tensor::<TestBackend, 1>::from_data(wrong_data, &Default::default());
    println!("Wrong tensor (2D data, 1D type) dims: {:?}", wrong_tensor.dims());
    println!("Wrong tensor data shape: {:?}", wrong_tensor.to_data().shape);
}
