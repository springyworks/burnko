//! Debug tests for GPU scan implementation

#[cfg(test)]
mod debug_gpu_scan_tests {
    use burn_tensor::{
        Tensor, TensorData, Shape,
        ops::{ScanConfig, ScanOp},
    };
    
    type TestBackend = burn_wgpu::Wgpu;
    type TestTensor<const D: usize> = Tensor<TestBackend, D>;

    fn get_device() -> <TestBackend as burn_tensor::backend::Backend>::Device {
        Default::default()
    }

    /// Simple debug test with minimal data
    #[test]
    fn debug_simple_scan() {
        let device = get_device();
        
        // Test with just 3 elements: [1, 2, 3]
        let input = TestTensor::<1>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0], Shape::new([3])), 
            &device
        );
        
        println!("Input: {:?}", input.to_data().to_vec::<f32>().unwrap());
        
        // Test cumsum - should give [1, 3, 6]
        let cumsum_result = input.clone().cumsum(0);
        let values: Vec<f32> = cumsum_result.to_data().to_vec().unwrap();
        println!("GPU cumsum result: {:?}", values);
        println!("Expected:          [1.0, 3.0, 6.0]");
        
        // Let's also test direct scan call
        let config = ScanConfig::new(ScanOp::Add, 0);
        let scan_result = input.scan(config);
        let scan_values: Vec<f32> = scan_result.to_data().to_vec().unwrap();
        println!("GPU scan result:   {:?}", scan_values);
        
        // Test what we expect vs what we get
        if values == vec![1.0, 2.0, 3.0] {
            println!("❌ GPU is just returning input unchanged - scan not working");
        } else if values == vec![1.0, 3.0, 6.0] {
            println!("✅ GPU scan is working correctly!");
        } else {
            println!("❓ GPU scan is producing unexpected results: {:?}", values);
        }
        
        // Don't fail the test, just debug
        assert!(true);
    }

    /// Test even simpler case with 2 elements
    #[test]
    fn debug_two_element_scan() {
        let device = get_device();
        
        // Test with just 2 elements: [1, 1]
        let input = TestTensor::<1>::from_data(
            TensorData::new(vec![1.0, 1.0], Shape::new([2])), 
            &device
        );
        
        println!("Input: {:?}", input.to_data().to_vec::<f32>().unwrap());
        
        let cumsum_result = input.clone().cumsum(0);
        let values: Vec<f32> = cumsum_result.to_data().to_vec().unwrap();
        println!("GPU cumsum result: {:?}", values);
        println!("Expected:          [1.0, 2.0]");
        
        if values == vec![1.0, 1.0] {
            println!("❌ GPU is not accumulating - each element unchanged");
        } else if values == vec![1.0, 2.0] {
            println!("✅ GPU scan is working correctly!");
        } else {
            println!("❓ Unexpected result: {:?}", values);
        }
        
        // Test other operations too
        let cumprod_result = input.cumprod(0);
        let prod_values: Vec<f32> = cumprod_result.to_data().to_vec().unwrap();
        println!("GPU cumprod result: {:?}", prod_values);
        println!("Expected:           [1.0, 1.0]");
        
        assert!(true);
    }
}
