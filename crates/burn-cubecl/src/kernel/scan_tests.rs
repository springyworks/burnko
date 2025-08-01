//! Tests for GPU scan operations

#[cfg(test)]
mod scan_tests {
    use burn_tensor::ops::{ScanConfig, ScanOp};

    /// Test basic functionality without GPU dependency
    #[test]
    fn test_scan_config_creation() {
        // Test that we can create scan configurations
        let config_add = ScanConfig::new(ScanOp::Add, 0);
        assert_eq!(config_add.operation, ScanOp::Add);
        assert_eq!(config_add.dim, 0);
        
        let config_mul = ScanConfig::new(ScanOp::Mul, 1);
        assert_eq!(config_mul.operation, ScanOp::Mul);
        assert_eq!(config_mul.dim, 1);
    }

    // Note: GPU-dependent tests would go here but are omitted for CI stability
    // They can be run manually with appropriate GPU backends available
}
