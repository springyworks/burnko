//! Comprehensive 2D GPU Performance Test
//! Tests both scan and FFT operations for WGPU backend performance

#[cfg(test)]
mod gpu_2d_performance_tests {
    use burn_tensor::{
        Tensor, TensorData, Shape, Distribution,
        ops::{ScanConfig, ScanOp},
    };
    use std::time::{Duration, Instant};
    
    // Use burn-wgpu backend for GPU testing
    type TestBackend = burn_wgpu::Wgpu;
    type TestTensor<const D: usize> = Tensor<TestBackend, D>;

    fn get_device() -> <TestBackend as burn_tensor::backend::Backend>::Device {
        Default::default()
    }

    #[test]
    fn test_2d_scan_performance() {
        println!("\n🔥🔥🔥 2D SCAN GPU PERFORMANCE TEST 🔥🔥🔥");
        
        let device = get_device();
        
        // Test different 2D tensor sizes
        let sizes = vec![
            (64, 64),     // 4K elements
            (128, 128),   // 16K elements
            (256, 256),   // 64K elements  
            (512, 512),   // 256K elements
            (1024, 1024), // 1M elements
        ];
        
        for (rows, cols) in sizes {
            println!("\n📊 Testing {}×{} matrix ({} elements)...", rows, cols, rows * cols);
            
            // Create test tensor
            let values: Vec<f32> = (0..rows*cols)
                .map(|i| (i as f32 * 0.001) % 10.0 + 1.0)
                .collect();
            let tensor = TestTensor::<2>::from_data(
                TensorData::new(values, Shape::new([rows, cols])), 
                &device
            );
            
            // Test scan along axis 0 (down columns)
            let start_time = Instant::now();
            let config_axis0 = ScanConfig::new(ScanOp::Add, 0);
            let result_axis0 = tensor.clone().scan(config_axis0);
            let _values_axis0: Vec<f32> = result_axis0.to_data().to_vec().unwrap();
            let axis0_time = start_time.elapsed();
            
            // Test scan along axis 1 (across rows)
            let start_time = Instant::now();
            let config_axis1 = ScanConfig::new(ScanOp::Add, 1);
            let result_axis1 = tensor.scan(config_axis1);
            let _values_axis1: Vec<f32> = result_axis1.to_data().to_vec().unwrap();
            let axis1_time = start_time.elapsed();
            
            // Performance metrics
            let axis0_throughput = (rows * cols) as f64 / axis0_time.as_secs_f64() / 1_000_000.0;
            let axis1_throughput = (rows * cols) as f64 / axis1_time.as_secs_f64() / 1_000_000.0;
            
            println!("   Axis 0 (columns): {:?} ({:.2} Melems/sec)", axis0_time, axis0_throughput);
            println!("   Axis 1 (rows):    {:?} ({:.2} Melems/sec)", axis1_time, axis1_throughput);
            
            // Performance expectations
            if axis0_throughput > 50.0 || axis1_throughput > 50.0 {
                println!("   🚀 EXCELLENT GPU performance!");
            } else if axis0_throughput > 10.0 || axis1_throughput > 10.0 {
                println!("   ✅ Good GPU performance");
            } else {
                println!("   ⚠️  GPU underutilized");
            }
        }
        
        println!("\n✅ 2D Scan GPU performance test completed!");
    }

    #[test]
    fn test_2d_scan_correctness() {
        println!("\n🔍 2D SCAN CORRECTNESS VERIFICATION");
        
        let device = get_device();
        
        // Simple 3×4 test matrix
        let values = vec![
            1.0, 2.0, 3.0, 4.0,  // Row 0
            5.0, 6.0, 7.0, 8.0,  // Row 1
            9.0, 10.0, 11.0, 12.0, // Row 2
        ];
        let tensor = TestTensor::<2>::from_data(
            TensorData::new(values, Shape::new([3, 4])), 
            &device
        );
        
        println!("Input matrix (3×4):");
        println!("[[1, 2, 3, 4],");
        println!(" [5, 6, 7, 8],");
        println!(" [9, 10, 11, 12]]");
        
        // Test cumsum along axis 0 (down columns)
        let config_axis0 = ScanConfig::new(ScanOp::Add, 0);
        let result_axis0 = tensor.clone().scan(config_axis0);
        let values_axis0: Vec<f32> = result_axis0.to_data().to_vec().unwrap();
        
        println!("\nCumsum axis 0 (down columns):");
        println!("Expected: [[1, 2, 3, 4], [6, 8, 10, 12], [15, 18, 21, 24]]");
        println!("Actual:   {:?}", values_axis0);
        
        let expected_axis0 = vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 21.0, 24.0];
        assert_eq!(values_axis0, expected_axis0, "Axis 0 cumsum failed");
        
        // Test cumsum along axis 1 (across rows)
        let config_axis1 = ScanConfig::new(ScanOp::Add, 1);
        let result_axis1 = tensor.scan(config_axis1);
        let values_axis1: Vec<f32> = result_axis1.to_data().to_vec().unwrap();
        
        println!("\nCumsum axis 1 (across rows):");
        println!("Expected: [[1, 3, 6, 10], [5, 11, 18, 26], [9, 19, 30, 42]]");
        println!("Actual:   {:?}", values_axis1);
        
        let expected_axis1 = vec![1.0, 3.0, 6.0, 10.0, 5.0, 11.0, 18.0, 26.0, 9.0, 19.0, 30.0, 42.0];
        assert_eq!(values_axis1, expected_axis1, "Axis 1 cumsum failed");
        
        println!("\n✅ 2D scan correctness verified!");
    }

    #[test]
    fn test_2d_mixed_operations() {
        println!("\n🌟 2D MIXED SCAN OPERATIONS TEST");
        
        let device = get_device();
        
        // Create a 4×4 test matrix
        let values = vec![
            1.0, 2.0, 3.0, 4.0,
            2.0, 3.0, 4.0, 5.0,
            3.0, 4.0, 5.0, 6.0,
            4.0, 5.0, 6.0, 7.0,
        ];
        let tensor = TestTensor::<2>::from_data(
            TensorData::new(values, Shape::new([4, 4])), 
            &device
        );
        
        // Test different scan operations
        let operations = vec![
            (ScanOp::Add, "Add (cumsum)"),
            (ScanOp::Mul, "Mul (cumprod)"),
            (ScanOp::Max, "Max (cummax)"),
            (ScanOp::Min, "Min (cummin)"),
        ];
        
        for (op, name) in operations {
            println!("\n🔄 Testing {} operation:", name);
            
            for axis in 0..2 {
                let start_time = Instant::now();
                let config = ScanConfig::new(op, axis);
                let result = tensor.clone().scan(config);
                let _values: Vec<f32> = result.to_data().to_vec().unwrap();
                let duration = start_time.elapsed();
                
                let throughput = 16.0 / duration.as_secs_f64() / 1000.0; // 16 elements in Kelems/sec
                println!("   Axis {}: {:?} ({:.2} Kelems/sec)", axis, duration, throughput);
            }
        }
        
        println!("\n✅ Mixed operations completed!");
    }

    #[test] 
    #[ignore] // Enable when FFT GPU implementation is ready
    fn test_2d_fft_performance() {
        println!("\n🌊 2D FFT GPU PERFORMANCE TEST");
        
        let device = get_device();
        
        // Test different FFT sizes (powers of 2 for optimal performance)
        let sizes = vec![64, 128, 256, 512];
        
        for size in sizes {
            println!("\n📊 Testing {}×{} FFT ({} elements)...", size, size, size * size);
            
            // Create test tensor with some frequency content
            let tensor: TestTensor<2> = Tensor::random(
                [size, size],
                Distribution::Uniform(-1.0, 1.0),
                &device,
            );
            
            // Test FFT along different dimensions
            for dim in 0..2 {
                let start_time = Instant::now();
                let fft_result = tensor.clone().fft(dim);
                let _complex_values: Vec<f32> = fft_result.to_data().to_vec().unwrap();
                let fft_time = start_time.elapsed();
                
                // Test IFFT
                let start_time = Instant::now();
                let ifft_result = fft_result.ifft(dim);
                let _recovered_values: Vec<f32> = ifft_result.to_data().to_vec().unwrap();
                let ifft_time = start_time.elapsed();
                
                let fft_throughput = (size * size) as f64 / fft_time.as_secs_f64() / 1_000_000.0;
                let ifft_throughput = (size * size) as f64 / ifft_time.as_secs_f64() / 1_000_000.0;
                
                println!("   Dim {} FFT:  {:?} ({:.2} Melems/sec)", dim, fft_time, fft_throughput);
                println!("   Dim {} IFFT: {:?} ({:.2} Melems/sec)", dim, ifft_time, ifft_throughput);
            }
        }
        
        println!("\n✅ 2D FFT GPU performance test completed!");
    }

    #[test]
    fn test_gpu_utilization_2d() {
        println!("\n⚡ GPU UTILIZATION WITH 2D OPERATIONS");
        
        let device = get_device();
        
        // Create large 2D tensors to maximize GPU usage
        let sizes = vec![(512, 1024), (1024, 512), (1024, 1024)];
        
        for (rows, cols) in sizes {
            println!("\n🔥 Testing {}×{} matrix...", rows, cols);
            
            let tensor: TestTensor<2> = Tensor::random(
                [rows, cols],
                Distribution::Uniform(0.0, 1.0),
                &device,
            );
            
            let start_time = Instant::now();
            
            // Perform multiple operations to stress GPU
            for _ in 0..5 {
                for axis in 0..2 {
                    for &op in &[ScanOp::Add, ScanOp::Mul, ScanOp::Max] {
                        let config = ScanConfig::new(op, axis);
                        let _result = tensor.clone().scan(config);
                        let _values: Vec<f32> = _result.to_data().to_vec().unwrap();
                    }
                }
            }
            
            let total_time = start_time.elapsed();
            let total_ops = 5 * 2 * 3; // 5 iterations × 2 axes × 3 operations
            let ops_per_sec = total_ops as f64 / total_time.as_secs_f64();
            
            println!("   {} operations in {:?} ({:.1} ops/sec)", total_ops, total_time, ops_per_sec);
            
            if ops_per_sec > 50.0 {
                println!("   🚀 Excellent GPU throughput!");
            } else if ops_per_sec > 20.0 {
                println!("   ✅ Good GPU performance");
            } else {
                println!("   ⚠️  Low GPU utilization");
            }
        }
        
        println!("\n✅ GPU utilization test completed!");
    }
}
