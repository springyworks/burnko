#[cfg(test)]
mod scan_tests {
    use burn_tensor::{
        Tensor, TensorData, Shape,
        ops::{ScanOp, ScanConfig},
    };
    
    type TestBackend = crate::NdArray<f32>;
    type TestTensor<const D: usize> = Tensor<TestBackend, D>;

    /// Test NdArray backend cumulative sum operations
    #[test]
    fn test_ndarray_cumsum() {
        let device = Default::default();
        
        // 1D test
        let tensor_1d = TestTensor::<1>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new([5])), 
            &device
        );
        let result_1d = tensor_1d.cumsum(0);
        let values: Vec<f32> = result_1d.to_data().to_vec().unwrap();
        assert_eq!(values, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
        
        // 2D test
        let tensor_2d = TestTensor::<2>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new([2, 3])), 
            &device
        );
        
        // Cumsum along dimension 0
        let result_2d_dim0 = tensor_2d.clone().cumsum(0);
        let values_dim0: Vec<f32> = result_2d_dim0.to_data().to_vec().unwrap();
        assert_eq!(values_dim0, vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
        
        // Cumsum along dimension 1
        let result_2d_dim1 = tensor_2d.cumsum(1);
        let values_dim1: Vec<f32> = result_2d_dim1.to_data().to_vec().unwrap();
        assert_eq!(values_dim1, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    }

    /// Test NdArray backend cumulative product operations
    #[test]
    fn test_ndarray_cumprod() {
        let device = Default::default();
        
        let tensor = TestTensor::<1>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new([5])), 
            &device
        );
        let result = tensor.cumprod(0);
        let values: Vec<f32> = result.to_data().to_vec().unwrap();
        assert_eq!(values, vec![1.0, 2.0, 6.0, 24.0, 120.0]);
    }

    /// Test NdArray backend cumulative maximum operations
    #[test]
    fn test_ndarray_cummax() {
        let device = Default::default();
        
        let tensor = TestTensor::<1>::from_data(
            TensorData::new(vec![1.0, 5.0, 3.0, 7.0, 2.0], Shape::new([5])), 
            &device
        );
        let result = tensor.cummax(0);
        let values: Vec<f32> = result.to_data().to_vec().unwrap();
        assert_eq!(values, vec![1.0, 5.0, 5.0, 7.0, 7.0]);
    }

    /// Test NdArray backend cumulative minimum operations
    #[test]
    fn test_ndarray_cummin() {
        let device = Default::default();
        
        let tensor = TestTensor::<1>::from_data(
            TensorData::new(vec![5.0, 1.0, 3.0, 0.5, 2.0], Shape::new([5])), 
            &device
        );
        let result = tensor.cummin(0);
        let values: Vec<f32> = result.to_data().to_vec().unwrap();
        assert_eq!(values, vec![5.0, 1.0, 1.0, 0.5, 0.5]);
    }

    /// Test NdArray backend general scan operations
    #[test]
    fn test_ndarray_scan_operations() {
        let device = Default::default();
        
        let tensor = TestTensor::<1>::from_data(
            TensorData::new(vec![2.0, 3.0, 1.0, 4.0], Shape::new([4])), 
            &device
        );
        
        // Test Add scan
        let config_add = ScanConfig::new(ScanOp::Add, 0);
        let result_add = tensor.clone().scan(config_add);
        let values_add: Vec<f32> = result_add.to_data().to_vec().unwrap();
        assert_eq!(values_add, vec![2.0, 5.0, 6.0, 10.0]);
        
        // Test Mul scan
        let config_mul = ScanConfig::new(ScanOp::Mul, 0);
        let result_mul = tensor.clone().scan(config_mul);
        let values_mul: Vec<f32> = result_mul.to_data().to_vec().unwrap();
        assert_eq!(values_mul, vec![2.0, 6.0, 6.0, 24.0]);
        
        // Test Max scan
        let config_max = ScanConfig::new(ScanOp::Max, 0);
        let result_max = tensor.clone().scan(config_max);
        let values_max: Vec<f32> = result_max.to_data().to_vec().unwrap();
        assert_eq!(values_max, vec![2.0, 3.0, 3.0, 4.0]);
        
        // Test Min scan
        let config_min = ScanConfig::new(ScanOp::Min, 0);
        let result_min = tensor.scan(config_min);
        let values_min: Vec<f32> = result_min.to_data().to_vec().unwrap();
        assert_eq!(values_min, vec![2.0, 2.0, 1.0, 1.0]);
    }

    /// Test scan configuration
    #[test]
    fn test_scan_config() {
        // Test ScanConfig creation and properties
        let config_add = ScanConfig::new(ScanOp::Add, 0);
        assert_eq!(config_add.op, ScanOp::Add);
        assert_eq!(config_add.dim, 0);
        assert_eq!(config_add.inclusive, true);
        
        let config_mul = ScanConfig::new(ScanOp::Mul, 1);
        assert_eq!(config_mul.op, ScanOp::Mul);
        assert_eq!(config_mul.dim, 1);
        assert_eq!(config_mul.inclusive, true);
    }

    /// Test analytical cases with well-known mathematical properties
    #[test]
    fn test_analytical_scan_cases() {
        let device = Default::default();
        
        // Test 1: All ones - cumsum should be [1, 2, 3, 4, 5, ...]
        let ones = TestTensor::<1>::from_data(
            TensorData::new(vec![1.0; 5], Shape::new([5])), 
            &device
        );
        let cumsum_ones = ones.clone().cumsum(0);
        let values: Vec<f32> = cumsum_ones.to_data().to_vec().unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        
        // Test 2: All ones - cumprod should remain all ones
        let cumprod_ones = ones.cumprod(0);
        let values: Vec<f32> = cumprod_ones.to_data().to_vec().unwrap();
        assert_eq!(values, vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        
        // Test 3: All zeros - cumsum should remain all zeros
        let zeros = TestTensor::<1>::from_data(
            TensorData::new(vec![0.0; 5], Shape::new([5])), 
            &device
        );
        let cumsum_zeros = zeros.cumsum(0);
        let values: Vec<f32> = cumsum_zeros.to_data().to_vec().unwrap();
        assert_eq!(values, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        
        // Test 4: Powers of 2 - known sequence [1, 2, 4, 8, 16]
        let powers_of_2 = TestTensor::<1>::from_data(
            TensorData::new(vec![1.0, 2.0, 4.0, 8.0, 16.0], Shape::new([5])), 
            &device
        );
        let cumsum_powers = powers_of_2.cumsum(0);
        let values: Vec<f32> = cumsum_powers.to_data().to_vec().unwrap();
        assert_eq!(values, vec![1.0, 3.0, 7.0, 15.0, 31.0]); // 2^n - 1 pattern
        
        // Test 5: Alternating signs - cumsum should oscillate
        let alternating = TestTensor::<1>::from_data(
            TensorData::new(vec![1.0, -1.0, 1.0, -1.0, 1.0], Shape::new([5])), 
            &device
        );
        let cumsum_alt = alternating.cumsum(0);
        let values: Vec<f32> = cumsum_alt.to_data().to_vec().unwrap();
        assert_eq!(values, vec![1.0, 0.0, 1.0, 0.0, 1.0]);
        
        // Test 6: Decreasing sequence for cummax (should be constant)
        let decreasing = TestTensor::<1>::from_data(
            TensorData::new(vec![10.0, 8.0, 6.0, 4.0, 2.0], Shape::new([5])), 
            &device
        );
        let cummax_dec = decreasing.cummax(0);
        let values: Vec<f32> = cummax_dec.to_data().to_vec().unwrap();
        assert_eq!(values, vec![10.0, 10.0, 10.0, 10.0, 10.0]);
        
        // Test 7: Increasing sequence for cummin (should be constant)
        let increasing = TestTensor::<1>::from_data(
            TensorData::new(vec![2.0, 4.0, 6.0, 8.0, 10.0], Shape::new([5])), 
            &device
        );
        let cummin_inc = increasing.cummin(0);
        let values: Vec<f32> = cummin_inc.to_data().to_vec().unwrap();
        assert_eq!(values, vec![2.0, 2.0, 2.0, 2.0, 2.0]);
    }

    /// Test edge cases and boundary conditions
    #[test]
    fn test_scan_edge_cases() {
        let device = Default::default();
        
        // Test 1: Single element tensor
        let single = TestTensor::<1>::from_data(
            TensorData::new(vec![42.0], Shape::new([1])), 
            &device
        );
        let cumsum_single = single.cumsum(0);
        let values: Vec<f32> = cumsum_single.to_data().to_vec().unwrap();
        assert_eq!(values, vec![42.0]);
        
        // Test 2: Very small values (numerical stability)
        let tiny = TestTensor::<1>::from_data(
            TensorData::new(vec![1e-10, 1e-10, 1e-10], Shape::new([3])), 
            &device
        );
        let cumsum_tiny = tiny.cumsum(0);
        let values: Vec<f32> = cumsum_tiny.to_data().to_vec().unwrap();
        assert!((values[0] - 1e-10).abs() < 1e-15);
        assert!((values[1] - 2e-10).abs() < 1e-15);
        assert!((values[2] - 3e-10).abs() < 1e-15);
        
        // Test 3: Mixed positive/negative extremes
        let extremes = TestTensor::<1>::from_data(
            TensorData::new(vec![f32::MAX / 2.0, -f32::MAX / 2.0, f32::MAX / 2.0], Shape::new([3])), 
            &device
        );
        let cumsum_extremes = extremes.cumsum(0);
        let values: Vec<f32> = cumsum_extremes.to_data().to_vec().unwrap();
        // Should handle large number arithmetic without overflow
        assert!(values[0].is_finite());
        assert!(values[1].is_finite());
        assert!(values[2].is_finite());
    }

    /// Test 2D tensor analytical cases with multi-dimensional operations
    #[test]
    fn test_2d_analytical_cases() {
        let device = Default::default();
        
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
        
        // Cumsum along columns (dim 0) - each column should accumulate
        let cumsum_cols = identity.cumsum(0);
        let values: Vec<f32> = cumsum_cols.to_data().to_vec().unwrap();
        assert_eq!(values, vec![
            1.0, 0.0, 0.0,  // [1, 0, 0]
            1.0, 1.0, 0.0,  // [1, 1, 0] - columns accumulate down
            1.0, 1.0, 1.0   // [1, 1, 1]
        ]);
        
        // Test constant matrix - all elements same value
        let constant_matrix = TestTensor::<2>::from_data(
            TensorData::new(vec![5.0; 6], Shape::new([2, 3])), 
            &device
        );
        
        // Cumsum should create predictable arithmetic progressions
        let cumsum_const = constant_matrix.cumsum(1);
        let values: Vec<f32> = cumsum_const.to_data().to_vec().unwrap();
        assert_eq!(values, vec![
            5.0, 10.0, 15.0,  // [5, 10, 15]
            5.0, 10.0, 15.0   // [5, 10, 15]
        ]);
    }

    /// Test mathematical properties and invariants
    #[test]
    fn test_scan_mathematical_properties() {
        let device = Default::default();
        
        // Test: cumsum of cumsum should equal triangular numbers pattern
        let sequence = TestTensor::<1>::from_data(
            TensorData::new(vec![1.0, 1.0, 1.0, 1.0], Shape::new([4])), 
            &device
        );
        
        let first_cumsum = sequence.clone().cumsum(0);  // [1, 2, 3, 4]
        let values1: Vec<f32> = first_cumsum.to_data().to_vec().unwrap();
        assert_eq!(values1, vec![1.0, 2.0, 3.0, 4.0]);
        
        // Second cumsum should give triangular numbers
        let second_cumsum = first_cumsum.cumsum(0);  // [1, 3, 6, 10]
        let values2: Vec<f32> = second_cumsum.to_data().to_vec().unwrap();
        assert_eq!(values2, vec![1.0, 3.0, 6.0, 10.0]); // Triangular numbers: n(n+1)/2
        
        // Test: cumprod with powers should follow exponential pattern
        let base_2 = TestTensor::<1>::from_data(
            TensorData::new(vec![2.0, 2.0, 2.0, 2.0], Shape::new([4])), 
            &device
        );
        let cumprod_2 = base_2.cumprod(0);
        let values: Vec<f32> = cumprod_2.to_data().to_vec().unwrap();
        assert_eq!(values, vec![2.0, 4.0, 8.0, 16.0]); // Powers of 2: 2^n
        
        // Test: cummax/cummin idempotency with sorted data
        let sorted_asc = TestTensor::<1>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new([5])), 
            &device
        );
        let cummax_sorted = sorted_asc.cummax(0);
        let values: Vec<f32> = cummax_sorted.to_data().to_vec().unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]); // Should be unchanged
        
        let sorted_desc = TestTensor::<1>::from_data(
            TensorData::new(vec![5.0, 4.0, 3.0, 2.0, 1.0], Shape::new([5])), 
            &device
        );
        let cummin_sorted = sorted_desc.cummin(0);
        let values: Vec<f32> = cummin_sorted.to_data().to_vec().unwrap();
        assert_eq!(values, vec![5.0, 4.0, 3.0, 2.0, 1.0]); // Should be unchanged
    }

    /// Test larger tensor sizes for performance and correctness
    #[test]
    fn test_large_tensor_scan() {
        let device = Default::default();
        
        // Test with larger tensor to ensure scalability
        let size = 1000;
        let large_ones: Vec<f32> = vec![1.0; size];
        let large_tensor = TestTensor::<1>::from_data(
            TensorData::new(large_ones, Shape::new([size])), 
            &device
        );
        
        // Cumsum of 1000 ones should be [1, 2, 3, ..., 1000]
        let cumsum_large = large_tensor.cumsum(0);
        let values: Vec<f32> = cumsum_large.to_data().to_vec().unwrap();
        
        // Verify first few and last few values
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 2.0);
        assert_eq!(values[2], 3.0);
        assert_eq!(values[size - 3], (size - 2) as f32);
        assert_eq!(values[size - 2], (size - 1) as f32);
        assert_eq!(values[size - 1], size as f32);
        
        // Verify the arithmetic progression property
        for i in 0..size {
            assert_eq!(values[i], (i + 1) as f32);
        }
    }

    /// Test exclusive vs inclusive scan behavior (when implemented)
    #[test]
    fn test_scan_exclusive_vs_inclusive() {
        let device = Default::default();
        
        let tensor = TestTensor::<1>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new([4])), 
            &device
        );
        
        // Current implementation is inclusive scan
        let inclusive_config = ScanConfig::new(ScanOp::Add, 0);
        let inclusive_result = tensor.scan(inclusive_config);
        let inclusive_values: Vec<f32> = inclusive_result.to_data().to_vec().unwrap();
        
        // Inclusive scan: [1, 3, 6, 10]
        assert_eq!(inclusive_values, vec![1.0, 3.0, 6.0, 10.0]);
        
        // Note: Exclusive scan would be [0, 1, 3, 6] - identity element + prefix sums
        // This validates current behavior and provides framework for exclusive implementation
    }
}
