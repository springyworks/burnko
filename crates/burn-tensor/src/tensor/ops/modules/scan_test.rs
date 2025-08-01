use crate::{ops::modules::scan::{ScanConfig, ScanOp}, tensor::Shape, TensorData, backend::Backend, Float};
use alloc::vec;

/// Test scan functionality with basic operations
pub fn test_scan<B: Backend>() {
    let device = B::Device::default();
    
    // Test cumulative sum
    let data = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new([4]));
    let tensor = crate::Tensor::<B, 1, Float>::from_data(data, &device);
    let result = tensor.cumsum(0);
    
    // Expected: [1.0, 3.0, 6.0, 10.0]
    println!("Cumsum result: {:?}", result);
    
    // Test cumulative product  
    let data = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new([4]));
    let tensor = crate::Tensor::<B, 1, Float>::from_data(data, &device);
    let result = tensor.cumprod(0);
    
    // Expected: [1.0, 2.0, 6.0, 24.0]
    println!("Cumprod result: {:?}", result);
    
    // Test custom scan with multiplication
    let data = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new([4]));
    let tensor = crate::Tensor::<B, 1, Float>::from_data(data, &device);
    let config = ScanConfig::new(ScanOp::Mul, 0).inclusive(true);
    let result = tensor.scan(config);
    
    // Expected: [1.0, 2.0, 6.0, 24.0]
    println!("Custom scan result: {:?}", result);
}

/// Test 2D scan operations
pub fn test_scan_2d<B: Backend>() {
    let device = B::Device::default();
    
    // Test 2D cumulative sum along dimension 1
    let data = TensorData::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
        Shape::new([2, 3])
    );
    let tensor = crate::Tensor::<B, 2, Float>::from_data(data, &device);
    let result = tensor.cumsum(1);
    
    // Expected: [[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]]
    println!("2D Cumsum result: {:?}", result);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scan_config_basic() {
        // Test ScanConfig creation and properties
        let config = ScanConfig::new(ScanOp::Add, 0).inclusive(true);
        assert_eq!(config.op, ScanOp::Add);
        assert_eq!(config.dim, 0);
        assert_eq!(config.inclusive, true);
        
        let config2 = ScanConfig::new(ScanOp::Mul, 1).inclusive(false);
        assert_eq!(config2.op, ScanOp::Mul);
        assert_eq!(config2.dim, 1);
        assert_eq!(config2.inclusive, false);
    }
    
    #[test]
    fn test_scan_basic() {
        // This test would run with the NdArray backend for example
        // test_scan::<burn_ndarray::NdArray>();
        // test_scan_2d::<burn_ndarray::NdArray>();
    }
}
