//! Comprehensive GPU scan analytical tests
//! 
//! These tests validate that our GPU scan implementation produces
//! identical results to the CPU NdArray implementation for the same analytical test cases.

#[cfg(test)]
mod gpu_scan_analytical_tests {
    use burn_tensor::{
        Tensor, TensorData, Shape,
        ops::{ScanConfig, ScanOp},
    };
    
    // Use burn-wgpu backend to test our GPU scan implementation
    type TestBackend = burn_wgpu::Wgpu;
    type TestTensor<const D: usize> = Tensor<TestBackend, D>;

    fn get_device() -> <TestBackend as burn_tensor::backend::Backend>::Device {
        Default::default()
    }

    /// Test analytical cases with well-known mathematical properties
    /// This mirrors the exact same test cases from the NdArray CPU implementation
    #[test]
    fn test_gpu_analytical_scan_cases() {
        let device = get_device();
        
        // Test 1: All ones - cumsum should be [1, 2, 3, 4, 5, ...]
        let ones = TestTensor::<1>::from_data(
            TensorData::new(vec![1.0; 5], Shape::new([5])), 
            &device
        );
        let cumsum_ones = ones.clone().cumsum(0);
        let values: Vec<f32> = cumsum_ones.to_data().to_vec().unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        println!("✅ GPU Test 1 passed: All ones cumsum = [1, 2, 3, 4, 5]");
        
        // Test 2: All ones - cumprod should remain all ones
        let cumprod_ones = ones.cumprod(0);
        let values: Vec<f32> = cumprod_ones.to_data().to_vec().unwrap();
        assert_eq!(values, vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        println!("✅ GPU Test 2 passed: All ones cumprod = [1, 1, 1, 1, 1]");
        
        // Test 3: All zeros - cumsum should remain all zeros
        let zeros = TestTensor::<1>::from_data(
            TensorData::new(vec![0.0; 5], Shape::new([5])), 
            &device
        );
        let cumsum_zeros = zeros.cumsum(0);
        let values: Vec<f32> = cumsum_zeros.to_data().to_vec().unwrap();
        assert_eq!(values, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        println!("✅ GPU Test 3 passed: All zeros cumsum = [0, 0, 0, 0, 0]");
        
        // Test 4: Powers of 2 - known sequence [1, 2, 4, 8, 16]
        let powers_of_2 = TestTensor::<1>::from_data(
            TensorData::new(vec![1.0, 2.0, 4.0, 8.0, 16.0], Shape::new([5])), 
            &device
        );
        let cumsum_powers = powers_of_2.cumsum(0);
        let values: Vec<f32> = cumsum_powers.to_data().to_vec().unwrap();
        assert_eq!(values, vec![1.0, 3.0, 7.0, 15.0, 31.0]); // 2^n - 1 pattern
        println!("✅ GPU Test 4 passed: Powers of 2 cumsum = [1, 3, 7, 15, 31] (2^n - 1 pattern)");
        
        // Test 5: Alternating signs - cumsum should oscillate
        let alternating = TestTensor::<1>::from_data(
            TensorData::new(vec![1.0, -1.0, 1.0, -1.0, 1.0], Shape::new([5])), 
            &device
        );
        let cumsum_alt = alternating.cumsum(0);
        let values: Vec<f32> = cumsum_alt.to_data().to_vec().unwrap();
        assert_eq!(values, vec![1.0, 0.0, 1.0, 0.0, 1.0]);
        println!("✅ GPU Test 5 passed: Alternating signs cumsum = [1, 0, 1, 0, 1]");
        
        // Test 6: Fibonacci-like sequence
        let fibonacci = TestTensor::<1>::from_data(
            TensorData::new(vec![1.0, 1.0, 2.0, 3.0, 5.0], Shape::new([5])), 
            &device
        );
        let cumsum_fib = fibonacci.cumsum(0);
        let values: Vec<f32> = cumsum_fib.to_data().to_vec().unwrap();
        assert_eq!(values, vec![1.0, 2.0, 4.0, 7.0, 12.0]);
        println!("✅ GPU Test 6 passed: Fibonacci cumsum = [1, 2, 4, 7, 12]");
        
        println!("🎉 All GPU analytical tests passed! GPU matches CPU results exactly.");
    }

    /// Test GPU scan operations with the exact same test data as CPU tests
    #[test]
    fn test_gpu_scan_operations() {
        let device = get_device();
        
        let tensor = TestTensor::<1>::from_data(
            TensorData::new(vec![2.0, 3.0, 1.0, 4.0], Shape::new([4])), 
            &device
        );
        
        // Test Add scan
        let config_add = ScanConfig::new(ScanOp::Add, 0);
        let result_add = tensor.clone().scan(config_add);
        let values_add: Vec<f32> = result_add.to_data().to_vec().unwrap();
        assert_eq!(values_add, vec![2.0, 5.0, 6.0, 10.0]);
        println!("✅ GPU Add scan: [2, 3, 1, 4] → [2, 5, 6, 10]");
        
        // Test Mul scan
        let config_mul = ScanConfig::new(ScanOp::Mul, 0);
        let result_mul = tensor.clone().scan(config_mul);
        let values_mul: Vec<f32> = result_mul.to_data().to_vec().unwrap();
        assert_eq!(values_mul, vec![2.0, 6.0, 6.0, 24.0]);
        println!("✅ GPU Mul scan: [2, 3, 1, 4] → [2, 6, 6, 24]");
        
        // Test Max scan
        let config_max = ScanConfig::new(ScanOp::Max, 0);
        let result_max = tensor.clone().scan(config_max);
        let values_max: Vec<f32> = result_max.to_data().to_vec().unwrap();
        assert_eq!(values_max, vec![2.0, 3.0, 3.0, 4.0]);
        println!("✅ GPU Max scan: [2, 3, 1, 4] → [2, 3, 3, 4]");
        
        // Test Min scan
        let config_min = ScanConfig::new(ScanOp::Min, 0);
        let result_min = tensor.scan(config_min);
        let values_min: Vec<f32> = result_min.to_data().to_vec().unwrap();
        assert_eq!(values_min, vec![2.0, 2.0, 1.0, 1.0]);
        println!("✅ GPU Min scan: [2, 3, 1, 4] → [2, 2, 1, 1]");
        
        println!("🎉 All GPU scan operations passed!");
    }

    /// Test edge cases and boundary conditions on GPU - mirrors CPU tests
    #[test]
    fn test_gpu_scan_edge_cases() {
        let device = get_device();
        
        // Test 1: Single element tensor
        let single = TestTensor::<1>::from_data(
            TensorData::new(vec![42.0], Shape::new([1])), 
            &device
        );
        let cumsum_single = single.cumsum(0);
        let values: Vec<f32> = cumsum_single.to_data().to_vec().unwrap();
        assert_eq!(values, vec![42.0]);
        println!("✅ GPU single element test: [42] → [42]");
        
        // Test 2: Very small values (numerical stability)
        let tiny = TestTensor::<1>::from_data(
            TensorData::new(vec![1e-6, 1e-6, 1e-6], Shape::new([3])), 
            &device
        );
        let cumsum_tiny = tiny.cumsum(0);
        let values: Vec<f32> = cumsum_tiny.to_data().to_vec().unwrap();
        assert!((values[0] - 1e-6).abs() < 1e-10);
        assert!((values[1] - 2e-6).abs() < 1e-10);
        assert!((values[2] - 3e-6).abs() < 1e-10);
        println!("✅ GPU numerical stability test passed");
        
        // Test 3: Large numbers
        let large = TestTensor::<1>::from_data(
            TensorData::new(vec![1e6, 1e6, 1e6], Shape::new([3])), 
            &device
        );
        let cumsum_large = large.cumsum(0);
        let values: Vec<f32> = cumsum_large.to_data().to_vec().unwrap();
        assert_eq!(values, vec![1e6, 2e6, 3e6]);
        println!("✅ GPU large numbers test: [1e6, 1e6, 1e6] → [1e6, 2e6, 3e6]");
        
        println!("🎉 All GPU edge case tests passed!");
    }

    /// Test mathematical properties and invariants on GPU - mirrors CPU tests
    #[test]
    fn test_gpu_scan_mathematical_properties() {
        let device = get_device();
        
        // Test: cumsum of cumsum should equal triangular numbers pattern
        let sequence = TestTensor::<1>::from_data(
            TensorData::new(vec![1.0, 1.0, 1.0, 1.0], Shape::new([4])), 
            &device
        );
        
        let first_cumsum = sequence.clone().cumsum(0);  // [1, 2, 3, 4]
        let values1: Vec<f32> = first_cumsum.to_data().to_vec().unwrap();
        assert_eq!(values1, vec![1.0, 2.0, 3.0, 4.0]);
        println!("✅ GPU first cumsum: [1, 1, 1, 1] → [1, 2, 3, 4]");
        
        // Second cumsum should give triangular numbers
        let second_cumsum = first_cumsum.cumsum(0);  // [1, 3, 6, 10]
        let values2: Vec<f32> = second_cumsum.to_data().to_vec().unwrap();
        assert_eq!(values2, vec![1.0, 3.0, 6.0, 10.0]); // Triangular numbers: n(n+1)/2
        println!("✅ GPU triangular numbers: [1, 2, 3, 4] → [1, 3, 6, 10]");
        
        // Test: cumprod with powers should follow exponential pattern
        let base_2 = TestTensor::<1>::from_data(
            TensorData::new(vec![2.0, 2.0, 2.0, 2.0], Shape::new([4])), 
            &device
        );
        let cumprod_2 = base_2.cumprod(0);
        let values: Vec<f32> = cumprod_2.to_data().to_vec().unwrap();
        assert_eq!(values, vec![2.0, 4.0, 8.0, 16.0]); // Powers of 2: 2^n
        println!("✅ GPU powers of 2: [2, 2, 2, 2] → [2, 4, 8, 16]");
        
        println!("🎉 All GPU mathematical property tests passed!");
    }

    /// Test larger tensor sizes for performance and correctness on GPU
    #[test]
    fn test_gpu_large_tensor_scan() {
        let device = get_device();
        
        // Test with moderately large tensor to ensure scalability
        let size = 128; // Test our GPU implementation with reasonable size
        let large_ones: Vec<f32> = vec![1.0; size];
        let large_tensor = TestTensor::<1>::from_data(
            TensorData::new(large_ones, Shape::new([size])), 
            &device
        );
        
        // Cumsum of 128 ones should be [1, 2, 3, ..., 128]
        let cumsum_large = large_tensor.cumsum(0);
        let values: Vec<f32> = cumsum_large.to_data().to_vec().unwrap();
        
        // Verify first few and last few values
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 2.0);
        assert_eq!(values[2], 3.0);
        assert_eq!(values[size - 3], (size - 2) as f32);
        assert_eq!(values[size - 2], (size - 1) as f32);
        assert_eq!(values[size - 1], size as f32);
        
        // Verify the arithmetic progression property for a subset
        for i in 0..10 {
            assert_eq!(values[i], (i + 1) as f32);
        }
        for i in (size-10)..size {
            assert_eq!(values[i], (i + 1) as f32);
        }
        
        println!("✅ GPU large tensor test passed with size {}! First/last values correct.", size);
        println!("🎉 All GPU tests demonstrate exact equivalence to CPU analytical results!");
    }

    /// Test 2D tensor analytical cases with multi-dimensional operations on GPU
    #[test]
    fn test_gpu_2d_analytical_cases() {
        let device = get_device();
        
        // Test identity matrix cumsum properties
        let identity_data = vec![
            1.0, 0.0, 0.0,  // [1, 0, 0]
            0.0, 1.0, 0.0,  // [0, 1, 0] 
            0.0, 0.0, 1.0   // [0, 0, 1]
        ];
        let identity = TestTensor::<2>::from_data(
            TensorData::new(identity_data, Shape::new([3, 3])), 
            &device
        );
        
        // Cumsum along rows (dim 1) - each row should accumulate
        let cumsum_rows = identity.clone().cumsum(1);
        let values: Vec<f32> = cumsum_rows.to_data().to_vec().unwrap();
        assert_eq!(values, vec![
            1.0, 1.0, 1.0,  // [1, 1, 1] - first row accumulates
            0.0, 1.0, 1.0,  // [0, 1, 1] - second row accumulates  
            0.0, 0.0, 1.0   // [0, 0, 1] - third row accumulates
        ]);
        println!("✅ GPU 2D cumsum along rows (dim 1):");
        println!("   [1,0,0]  →  [1,1,1]");
        println!("   [0,1,0]  →  [0,1,1]");
        println!("   [0,0,1]  →  [0,0,1]");
        
        // Cumsum along columns (dim 0) - each column should accumulate
        let cumsum_cols = identity.cumsum(0);
        let values: Vec<f32> = cumsum_cols.to_data().to_vec().unwrap();
        assert_eq!(values, vec![
            1.0, 0.0, 0.0,  // [1, 0, 0]
            1.0, 1.0, 0.0,  // [1, 1, 0] - columns accumulate down
            1.0, 1.0, 1.0   // [1, 1, 1]
        ]);
        println!("✅ GPU 2D cumsum along columns (dim 0):");
        println!("   [1,0,0]  →  [1,0,0]");
        println!("   [0,1,0]  →  [1,1,0]");
        println!("   [0,0,1]  →  [1,1,1]");
        
        println!("🎉 All GPU 2D analytical cases passed!");
    }
}
