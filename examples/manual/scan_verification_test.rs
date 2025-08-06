// Manual scan operation verification test
// This verifies that scan operations work properly without the testgen system

use burn_ndarray::{NdArray, NdArrayDevice};
use burn_tensor::{backend::Backend, Tensor, TensorData, Shape, Distribution};

type TestBackend = NdArray<f32>;

fn main() {
    let device = NdArrayDevice::default();
    
    println!("=== Burn Scan Operations Verification ===");
    
    // Test 1: Basic cumsum 1D
    println!("\n1. Testing basic cumsum 1D...");
    let data = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new([4]));
    let tensor = Tensor::<TestBackend, 1>::from_data(data.convert(), &device);
    let result = tensor.cumsum(0);
    let values = result.to_data().to_vec::<f32>().unwrap();
    println!("Input: [1, 2, 3, 4]");
    println!("Output: {:?}", values);
    println!("Expected: [1, 3, 6, 10]");
    assert_eq!(values, vec![1.0, 3.0, 6.0, 10.0]);
    println!("✓ Basic cumsum test passed!");
    
    // Test 2: Cumsum with analytical sequence (triangular numbers)
    println!("\n2. Testing analytical cumsum sequence...");
    let size = 1000;
    let data: Vec<f32> = (1..=size).map(|i| i as f32).collect();
    let tensor = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(data, Shape::new([size])).convert(), 
        &device
    );
    let result = tensor.cumsum(0);
    let values = result.to_data().to_vec::<f32>().unwrap();
    
    // Check a few key points: cumsum[i-1] should equal i*(i+1)/2 for 1-based indexing
    let test_points = [99, 499, 999]; // 0-based indices for 100th, 500th, 1000th elements
    for &idx in &test_points {
        let actual = values[idx];
        let expected = ((idx + 1) * (idx + 2) / 2) as f32;
        let error = (actual - expected).abs();
        println!("Index {}: actual={}, expected={}, error={}", idx, actual, expected, error);
        assert!(error < 1e-3, "Analytical sequence test failed at index {}", idx);
    }
    println!("✓ Analytical sequence test passed!");
    
    // Test 3: 2D cumsum
    println!("\n3. Testing 2D cumsum...");
    let data = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
    let tensor = Tensor::<TestBackend, 2>::from_data(data.convert(), &device);
    
    let result_dim0 = tensor.clone().cumsum(0);
    let values_dim0 = result_dim0.to_data().to_vec::<f32>().unwrap();
    println!("2D cumsum dim=0: {:?}", values_dim0);
    println!("Expected: [1, 2, 4, 6]");
    assert_eq!(values_dim0, vec![1.0, 2.0, 4.0, 6.0]);
    
    let result_dim1 = tensor.cumsum(1);
    let values_dim1 = result_dim1.to_data().to_vec::<f32>().unwrap();
    println!("2D cumsum dim=1: {:?}", values_dim1);
