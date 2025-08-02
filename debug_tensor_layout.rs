use burn_ndarray::{NdArray, NdArrayDevice};
use burn_tensor::backend::Backend;
use burn_tensor::{Tensor, TensorData, Shape};

fn main() {
    type TestBackend = NdArray<f32>;
    let device = NdArrayDevice::Cpu;
    
    // Create identity matrix
    let identity_data = vec![
        1.0, 0.0, 0.0,  // [1, 0, 0]
        0.0, 1.0, 0.0,  // [0, 1, 0] 
        0.0, 0.0, 1.0   // [0, 0, 1]
    ];
    let identity = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(identity_data.clone(), Shape::new([3, 3])), 
        &device
    );
    
    println!("Original identity matrix:");
    println!("Data (flattened): {:?}", identity_data);
    println!("Shape: {:?}", identity.shape());
    println!("As 2D:");
    for row in 0..3 {
        let row_data: Vec<f32> = (0..3).map(|col| identity_data[row * 3 + col]).collect();
        println!("  Row {}: {:?}", row, row_data);
    }
    
    // Test cumsum along dim 1 (rows)
    let cumsum_rows = identity.cumsum(1);
    let result_data: Vec<f32> = cumsum_rows.to_data().to_vec().unwrap();
    
    println!("\nCumsum along dim 1 (rows):");
    println!("Result (flattened): {:?}", result_data);
    println!("As 2D:");
    for row in 0..3 {
        let row_data: Vec<f32> = (0..3).map(|col| result_data[row * 3 + col]).collect();
        println!("  Row {}: {:?}", row, row_data);
    }
    
    // Expected for cumsum along rows:
    // Row 0: [1,0,0] -> [1, 1, 1]
    // Row 1: [0,1,0] -> [0, 1, 1] 
    // Row 2: [0,0,1] -> [0, 0, 1]
    // So flattened: [1,1,1, 0,1,1, 0,0,1]
    let expected = vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    println!("Expected: {:?}", expected);
    println!("Match: {}", result_data == expected);
}
