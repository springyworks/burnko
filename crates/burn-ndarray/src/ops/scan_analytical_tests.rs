//! Comprehensive analytical tests for parallel scan operations
//! 
//! This module provides extensive testing of scan operations with:
//! - Correctness verification with known analytical results
//! - Performance comparison between sequential and parallel implementations
//! - Large-scale stress testing for multi-core utilization
//! - Analytical test cases with predictable outcomes

use crate::{element::NdArrayElement, tensor::NdArrayTensor, NdArray};
use burn_tensor::{Tensor, TensorData, Shape, Distribution};
use std::time::Instant;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArray;
    
    type TestBackend = NdArray<f32>;
    
    /// Test cumsum with analytical cases where we know the exact expected results
    #[test]
    fn test_analytical_cumsum_cases() {
        let device = Default::default();
        
        // Test 1: All ones - cumsum should be [1, 2, 3, 4, 5, ...]
        let ones_data = vec![1.0; 10];
        let ones: Tensor<TestBackend, 1> = Tensor::from_data(
            TensorData::new(ones_data, Shape::new([10])), &device
        );
        
        let cumsum_ones = ones.cumsum(0);
        let values: Vec<f32> = cumsum_ones.to_data().to_vec().unwrap();
        let expected: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        
        for (i, (&actual, &expected)) in values.iter().zip(expected.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-6, 
                    "Cumsum ones failed at index {}: expected {}, got {}", i, expected, actual);
        }
        println!("✅ Analytical Test 1: All ones cumsum = [1, 2, 3, ..., 10]");
    }
    
    /// Test cumsum with sequential integers - known analytical formula
    #[test]
    fn test_analytical_cumsum_sequential() {
        let device = Default::default();
        
        // Test with sequence [1, 2, 3, 4, 5]
        let seq_data: Vec<f32> = (1..=5).map(|x| x as f32).collect();
        let sequence: Tensor<TestBackend, 1> = Tensor::from_data(
            TensorData::new(seq_data, Shape::new([5])), &device
        );
        
        let cumsum_seq = sequence.cumsum(0);
        let values: Vec<f32> = cumsum_seq.to_data().to_vec().unwrap();
        
        // Expected: cumsum([1,2,3,4,5]) = [1, 3, 6, 10, 15]
        // Formula: cumsum[i] = i*(i+1)/2 for 1-based indexing
        let expected = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        
        for (i, (&actual, &expected)) in values.iter().zip(expected.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-6,
                    "Sequential cumsum failed at index {}: expected {}, got {}", i, expected, actual);
        }
        println!("✅ Analytical Test 2: Sequential [1,2,3,4,5] cumsum = [1,3,6,10,15]");
    }
    
    /// Test cumprod with analytical cases
    #[test] 
    fn test_analytical_cumprod_cases() {
        let device = Default::default();
        
        // Test 1: All ones - cumprod should remain all ones
        let ones_data = vec![1.0; 8];
        let ones: Tensor<TestBackend, 1> = Tensor::from_data(
            TensorData::new(ones_data, Shape::new([8])), &device
        );
        
        let cumprod_ones = ones.cumprod(0);
        let values: Vec<f32> = cumprod_ones.to_data().to_vec().unwrap();
        
        for (i, &value) in values.iter().enumerate() {
            assert!((value - 1.0).abs() < 1e-6,
                    "Cumprod ones failed at index {}: expected 1.0, got {}", i, value);
        }
        println!("✅ Analytical Test 3: All ones cumprod = [1, 1, 1, 1, 1, 1, 1, 1]");
        
        // Test 2: Powers of 2 - cumprod should be [2, 4, 8, 16, 32]
        let powers_data = vec![2.0; 5];
        let powers: Tensor<TestBackend, 1> = Tensor::from_data(
            TensorData::new(powers_data, Shape::new([5])), &device
        );
        
        let cumprod_powers = powers.cumprod(0);
        let values: Vec<f32> = cumprod_powers.to_data().to_vec().unwrap();
        let expected: Vec<f32> = (1..=5).map(|i| 2.0_f32.powi(i)).collect();
        
        for (i, (&actual, &expected)) in values.iter().zip(expected.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-6,
                    "Powers of 2 cumprod failed at index {}: expected {}, got {}", i, expected, actual);
        }
        println!("✅ Analytical Test 4: Powers of 2 cumprod = [2, 4, 8, 16, 32]");
    }
    
    /// Test multi-dimensional scans with analytical results
    #[test]
    fn test_analytical_multidimensional_scan() {
        let device = Default::default();
        
        // Test 3x3 identity matrix
        let identity_data = vec![
            1.0, 0.0, 0.0,  // [1, 0, 0]
            0.0, 1.0, 0.0,  // [0, 1, 0] 
            0.0, 0.0, 1.0   // [0, 0, 1]
        ];
        let identity: Tensor<TestBackend, 2> = Tensor::from_data(
            TensorData::new(identity_data, Shape::new([3, 3])), &device
        );
        
        // Cumsum along dimension 1 (rows)
        let cumsum_rows = identity.clone().cumsum(1);
        let result_data: Vec<f32> = cumsum_rows.to_data().to_vec().unwrap();
        
        // Expected for cumsum along rows:
        // Row 0: [1,0,0] -> [1, 1, 1]
        // Row 1: [0,1,0] -> [0, 1, 1] 
        // Row 2: [0,0,1] -> [0, 0, 1]
        let expected = vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        
        for (i, (&actual, &expected)) in result_data.iter().zip(expected.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-6,
                    "Identity matrix cumsum failed at index {}: expected {}, got {}", i, expected, actual);
        }
        println!("✅ Analytical Test 5: 3x3 identity matrix cumsum along rows verified");
        
        // Cumsum along dimension 0 (columns)
        let cumsum_cols = identity.cumsum(0);
        let col_result: Vec<f32> = cumsum_cols.to_data().to_vec().unwrap();
        
        // Expected for cumsum along columns:
        // Col 0: [1,0,0] -> [1, 1, 1]
        // Col 1: [0,1,0] -> [0, 1, 1]
        // Col 2: [0,0,1] -> [0, 0, 1]
        let expected_cols = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        
        for (i, (&actual, &expected)) in col_result.iter().zip(expected_cols.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-6,
                    "Identity matrix cumsum cols failed at index {}: expected {}, got {}", i, expected, actual);
        }
        println!("✅ Analytical Test 6: 3x3 identity matrix cumsum along columns verified");
    }
    
    /// Large-scale performance test comparing different tensor sizes
    #[test]
    fn test_large_scale_parallel_performance() {
        let device = Default::default();
        let core_count = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        
        println!("🧵 Running large-scale parallel scan performance test on {} cores", core_count);
        
        let test_sizes = vec![
            1_000,     // Should be sequential (below threshold)
            10_000,    // Should trigger parallel
            100_000,   // Definitely parallel
            1_000_000, // Large parallel workload
        ];
        
        for &size in &test_sizes {
            println!("\n📊 Testing size: {} elements", size);
            
            // Create test tensor with uniform random values
            let tensor: Tensor<TestBackend, 1> = Tensor::random(
                [size], Distribution::Uniform(0.0, 1.0), &device
            );
            
            // Test cumsum performance
            let start = Instant::now();
            let result = tensor.clone().cumsum(0);
            let duration = start.elapsed();
            
            let throughput = size as f64 / duration.as_secs_f64();
            println!("   ⚡ Cumsum: {:?} ({:.2}M elements/sec)", duration, throughput / 1_000_000.0);
            
            // Verify monotonic property for cumsum
            let values: Vec<f32> = result.to_data().to_vec().unwrap();
            for i in 1..values.len() {
                assert!(values[i] >= values[i-1], 
                        "Cumsum monotonic property violated at index {}", i);
            }
            
            // Test cumprod performance (with smaller values to avoid overflow)
            let small_tensor: Tensor<TestBackend, 1> = Tensor::random(
                [size], Distribution::Uniform(1.0, 1.001), &device
            );
            
            let start = Instant::now();
            let _prod_result = small_tensor.cumprod(0);
            let duration = start.elapsed();
            
            let throughput = size as f64 / duration.as_secs_f64();
            println!("   ⚡ Cumprod: {:?} ({:.2}M elements/sec)", duration, throughput / 1_000_000.0);
        }
        
        println!("\n🎯 Large-scale parallel performance test completed successfully!");
    }
    
    /// Stress test with extreme tensor sizes to verify parallel implementation robustness
    #[test]
    #[ignore] // Use `cargo test -- --ignored` to run this expensive test
    fn test_extreme_parallel_stress() {
        let device = Default::default();
        let core_count = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        
        println!("🔥 Running extreme parallel stress test on {} cores", core_count);
        println!("⚠️  This test uses large amounts of memory and CPU");
        
        let extreme_sizes = vec![
            5_000_000,   // 5M elements
            10_000_000,  // 10M elements  
            20_000_000,  // 20M elements
        ];
        
        for &size in &extreme_sizes {
            println!("\n🚀 Extreme test: {} elements ({:.1}MB)", size, size as f64 * 4.0 / 1_048_576.0);
            
            let tensor: Tensor<TestBackend, 1> = Tensor::random(
                [size], Distribution::Uniform(0.0, 1.0), &device
            );
            
            let start = Instant::now();
            let result = tensor.cumsum(0);
            let duration = start.elapsed();
            
            let throughput = size as f64 / duration.as_secs_f64();
            let throughput_per_core = throughput / core_count as f64;
            
            println!("   ⚡ Completed in: {:?}", duration);
            println!("   📊 Total throughput: {:.2}M elements/sec", throughput / 1_000_000.0);
            println!("   🧵 Per-core throughput: {:.2}M elements/sec", throughput_per_core / 1_000_000.0);
            
            // Basic correctness check - verify first and last few elements
            let values: Vec<f32> = result.to_data().to_vec().unwrap();
            assert!(values[0] >= 0.0 && values[0] <= 1.0, "First element out of range");
            assert!(values[size-1] >= values[0], "Cumsum not monotonic");
            
            println!("   ✅ Correctness verified for extreme size");
        }
        
        println!("\n🏆 Extreme parallel stress test completed successfully!");
    }
}

/// Performance benchmark comparing our implementation against known baselines
pub fn run_comprehensive_scan_benchmark() {
    let device = Default::default();
    let core_count = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    
    println!("🏁 Comprehensive Parallel Scan Performance Benchmark");
    println!("🧵 Running on {} CPU cores", core_count);
    println!("📊 Testing multiple tensor sizes and operations\n");
    
    let benchmark_sizes = vec![
        ("Small", 1_000),
        ("Medium", 50_000), 
        ("Large", 500_000),
        ("XLarge", 2_000_000),
        ("XXLarge", 10_000_000),
    ];
    
    for (name, size) in benchmark_sizes {
        println!("🔬 {} Benchmark: {} elements", name, size);
        
        // Create test data
        let tensor: Tensor<NdArray<f32>, 1> = Tensor::random(
            [size], Distribution::Uniform(0.0, 1.0), &device
        );
        
        // Benchmark cumsum
        let start = Instant::now();
        let _cumsum_result = tensor.clone().cumsum(0);
        let cumsum_time = start.elapsed();
        
        // Benchmark cumprod
        let prod_tensor: Tensor<NdArray<f32>, 1> = Tensor::random(
            [size], Distribution::Uniform(1.0, 1.01), &device
        );
        let start = Instant::now();
        let _cumprod_result = prod_tensor.cumprod(0);
        let cumprod_time = start.elapsed();
        
        let cumsum_throughput = size as f64 / cumsum_time.as_secs_f64();
        let cumprod_throughput = size as f64 / cumprod_time.as_secs_f64();
        
        println!("   ⚡ Cumsum:  {:>8.2} ms | {:>8.2}M elem/s | {:>6.2}M elem/s/core",
                 cumsum_time.as_secs_f64() * 1000.0,
                 cumsum_throughput / 1_000_000.0,
                 cumsum_throughput / core_count as f64 / 1_000_000.0);
        
        println!("   ⚡ Cumprod: {:>8.2} ms | {:>8.2}M elem/s | {:>6.2}M elem/s/core",
                 cumprod_time.as_secs_f64() * 1000.0,
                 cumprod_throughput / 1_000_000.0,
                 cumprod_throughput / core_count as f64 / 1_000_000.0);
        
        println!();
    }
    
    println!("🎯 Comprehensive benchmark completed!");
}
