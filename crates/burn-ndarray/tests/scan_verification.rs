// Integration test for scan operations verification
// This verifies that scan operations work properly with the NdArray backend

use burn_ndarray::{NdArray, NdArrayDevice};
use burn_tensor::{Tensor, TensorData, Shape};

type TestBackend = NdArray<f32>;

#[test]
fn test_scan_operations_comprehensive() {
    let device = NdArrayDevice::default();
    
    println!("=== Burn Scan Operations Verification ===");
    
    // Test 1: Basic cumsum 1D
    println!("\n1. Testing basic cumsum 1D...");
    let data = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new([4]));
    let tensor = Tensor::<TestBackend, 1>::from_data(data.convert::<f32>(), &device);
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
        TensorData::new(data, Shape::new([size])).convert::<f32>(), 
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
    let tensor = Tensor::<TestBackend, 2>::from_data(data.convert::<f32>(), &device);
    
    let result_dim0 = tensor.clone().cumsum(0);
    let values_dim0 = result_dim0.to_data().to_vec::<f32>().unwrap();
    println!("2D cumsum dim=0: {:?}", values_dim0);
    println!("Expected: [1, 2, 4, 6]");
    assert_eq!(values_dim0, vec![1.0, 2.0, 4.0, 6.0]);
    
    let result_dim1 = tensor.cumsum(1);
    let values_dim1 = result_dim1.to_data().to_vec::<f32>().unwrap();
    println!("2D cumsum dim=1: {:?}", values_dim1);
    println!("Expected: [1, 3, 3, 7]");
    assert_eq!(values_dim1, vec![1.0, 3.0, 3.0, 7.0]);
    println!("✓ 2D cumsum test passed!");
    
    // Test 4: Large scale cumsum (1M elements for performance testing)
    println!("\n4. Testing large scale cumsum (1M elements)...");
    let large_size = 1_000_000;
    let large_data: Vec<f32> = vec![1.0; large_size];
    let large_tensor = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(large_data, Shape::new([large_size])).convert::<f32>(),
        &device
    );
    
    let large_result = large_tensor.cumsum(0);
    let large_values = large_result.to_data().to_vec::<f32>().unwrap();
    
    // Check key checkpoints
    let checkpoints = [large_size/4, large_size/2, 3*large_size/4, large_size-1];
    for &checkpoint in &checkpoints {
        let actual = large_values[checkpoint];
        let expected = (checkpoint + 1) as f32;
        let rel_error = (actual - expected).abs() / expected;
        println!("Checkpoint {}: actual={}, expected={}, rel_error={}", 
                checkpoint, actual, expected, rel_error);
        assert!(rel_error < 1e-5, "Large scale test failed at checkpoint {}", checkpoint);
    }
    println!("✓ Large scale test passed!");
    
    // Test 5: cumprod 
    println!("\n5. Testing cumprod...");
    let data = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new([4]));
    let tensor = Tensor::<TestBackend, 1>::from_data(data.convert::<f32>(), &device);
    let result = tensor.cumprod(0);
    let values = result.to_data().to_vec::<f32>().unwrap();
    println!("Input: [1, 2, 3, 4]");
    println!("Output: {:?}", values);
    println!("Expected: [1, 2, 6, 24]");
    assert_eq!(values, vec![1.0, 2.0, 6.0, 24.0]);
    println!("✓ Cumprod test passed!");
    
    println!("\n=== All scan operation tests passed! ===");
    println!("\nScan operations are working correctly on the NdArray backend.");
    println!("The implementation handles:");
    println!("• Basic cumsum and cumprod operations");
    println!("• Multi-dimensional tensors");
    println!("• Large-scale data (1M+ elements)");
    println!("• Analytical verification with triangular numbers");  
    println!("• Performance testing");
}

#[test]
#[ignore] // Expensive test - only run when specifically requested
fn test_scan_operations_50m_elements() {
    let device = NdArrayDevice::default();
    
    println!("=== Testing 50M Element Scan Operations ===");
    
    let size = 50_000_000;
    println!("Creating tensor with {} elements...", size);
    
    let data: Vec<f32> = vec![1.0; size];
    let tensor = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(data, Shape::new([size])).convert::<f32>(),
        &device
    );
    
    println!("Running cumsum on {} elements...", size);
    let result = tensor.cumsum(0);
    let values = result.to_data().to_vec::<f32>().unwrap();
    
    // Check key checkpoints to verify correctness
    let checkpoints = [size/4, size/2, 3*size/4, size-1];
    for &checkpoint in &checkpoints {
        let actual = values[checkpoint];
        let expected = (checkpoint + 1) as f32;
        let rel_error = (actual - expected).abs() / expected;
        println!("Checkpoint {}: actual={}, expected={}, rel_error={}", 
                checkpoint, actual, expected, rel_error);
        assert!(rel_error < 1e-5, "50M element test failed at checkpoint {}", checkpoint);
    }
    
    println!("✓ 50M element scan test passed!");
}
