use burn_ndarray::NdArray;
use burn_tensor::{backend::Backend, Tensor, TensorData, Shape};

fn main() {
    type TestBackend = NdArray<f32>;
    let device = Default::default();
    
    // Simple test: [1, 1, 1, 1, 1] should become [1, 2, 3, 4, 5]
    let data = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let tensor: Tensor<TestBackend, 1> = Tensor::from_data(
        TensorData::new(data.clone(), Shape::new([5])), &device
    );
    
    println!("Input: {:?}", data);
    
    let result = tensor.cumsum(0);
    let output: Vec<f32> = result.to_data().to_vec().unwrap();
    
    println!("Output: {:?}", output);
    println!("Expected: [1.0, 2.0, 3.0, 4.0, 5.0]");
    
    // Test with 2D tensor to see how axis iteration works
    let data_2d = vec![
        1.0, 1.0, 1.0,  // Row 0
        1.0, 1.0, 1.0,  // Row 1  
        1.0, 1.0, 1.0,  // Row 2
    ];
    let tensor_2d: Tensor<TestBackend, 2> = Tensor::from_data(
        TensorData::new(data_2d, Shape::new([3, 3])), &device
    );
    
    println!("\n2D Input:");
    println!("[[1, 1, 1],");
    println!(" [1, 1, 1],");
    println!(" [1, 1, 1]]");
    
    // Cumsum along dim 0 (columns)
    let result_cols = tensor_2d.clone().cumsum(0);
    let output_cols: Vec<f32> = result_cols.to_data().to_vec().unwrap();
    println!("\nCumsum along dim 0 (columns): {:?}", output_cols);
    println!("Expected: [1,1,1, 2,2,2, 3,3,3]");
    
    // Cumsum along dim 1 (rows)  
    let result_rows = tensor_2d.cumsum(1);
    let output_rows: Vec<f32> = result_rows.to_data().to_vec().unwrap();
    println!("\nCumsum along dim 1 (rows): {:?}", output_rows);
    println!("Expected: [1,2,3, 1,2,3, 1,2,3]");
}
