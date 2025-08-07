use burn::prelude::*;
use burn_ndarray::NdArray;
use burn_wgpu::{Wgpu, WgpuDevice};

/// Cross-backend comparison tests for scan and FFT operations
/// This module demonstrates how to use Burn's automatic test generation
/// system to compare performance and correctness across backends.

pub type NdArrayBackend = NdArray<f32>;
pub type WgpuBackend = Wgpu<f32, i32>;

/// Test configuration for cross-backend comparisons
pub struct CrossBackendTest {
    pub size: usize,
    pub iterations: usize,
}

impl Default for CrossBackendTest {
    fn default() -> Self {
        Self {
            size: 1024,
            iterations: 100,
        }
    }
}

/// Run scan operation benchmarks across backends
pub fn benchmark_scan_operations() {
    println!("=== Cross-Backend Scan Operation Benchmarks ===");
    
    let test_config = CrossBackendTest::default();
    
    // Test data
    let data: Vec<f32> = (0..test_config.size).map(|i| i as f32).collect();
    
    // NdArray backend test
    println!("\n🔄 Testing NdArray backend:");
    let ndarray_device = burn_ndarray::NdArrayDevice::Cpu;
    let ndarray_tensor = Tensor::<NdArrayBackend, 1>::from_floats(data.as_slice(), &ndarray_device);
    
    let start = std::time::Instant::now();
    for _ in 0..test_config.iterations {
        let _ = ndarray_tensor.clone().cumsum(0);
    }
    let ndarray_time = start.elapsed();
    
    println!("  ✅ Cumulative sum: {:.2}ms avg", ndarray_time.as_millis() as f64 / test_config.iterations as f64);
    
    // WGPU backend test (if available)
    let wgpu_device = WgpuDevice::default();
    println!("\n🚀 Testing WGPU backend:");
    let wgpu_tensor = Tensor::<WgpuBackend, 1>::from_floats(data.as_slice(), &wgpu_device);
        
        let start = std::time::Instant::now();
        for _ in 0..test_config.iterations {
            let _ = wgpu_tensor.clone().cumsum(0);
        }
        let wgpu_time = start.elapsed();
        
        println!("  ✅ Cumulative sum: {:.2}ms avg", wgpu_time.as_millis() as f64 / test_config.iterations as f64);
        
        // Performance comparison
        let speedup = ndarray_time.as_nanos() as f64 / wgpu_time.as_nanos() as f64;
        println!("\n📊 Performance Comparison:");
        println!("  WGPU is {:.2}x {} than NdArray", 
                 if speedup > 1.0 { speedup } else { 1.0 / speedup },
                 if speedup > 1.0 { "faster" } else { "slower" });
}
}

/// Run FFT operation benchmarks across backends
pub fn benchmark_fft_operations() {
    println!("\n=== Cross-Backend FFT Operation Benchmarks ===");
    
    let test_config = CrossBackendTest { size: 256, ..Default::default() };
    
    // Create 2D test data (square matrix)
    let mut data = Vec::new();
    for i in 0..test_config.size {
        for j in 0..test_config.size {
            data.push((i + j) as f32);
        }
    }
    
    // NdArray backend test
    println!("\n🔄 Testing NdArray backend:");
    let ndarray_device = burn_ndarray::NdArrayDevice::Cpu;
    let ndarray_tensor = Tensor::<NdArrayBackend, 2>::from_floats(
        data.as_slice(), &ndarray_device
    ).reshape([test_config.size, test_config.size]);
    
    let start = std::time::Instant::now();
    for _ in 0..test_config.iterations {
        let _ = ndarray_tensor.clone().fft2(0, 1);
    }
    let ndarray_time = start.elapsed();
    
    println!("  ✅ 2D FFT: {:.2}ms avg", ndarray_time.as_millis() as f64 / test_config.iterations as f64);
    
    // WGPU backend test (if available)
    let wgpu_device = WgpuDevice::default();
    println!("\n🚀 Testing WGPU backend:");
    let wgpu_tensor = Tensor::<WgpuBackend, 2>::from_floats(
        data.as_slice(), &wgpu_device
    ).reshape([test_config.size, test_config.size]);
    
    let start = std::time::Instant::now();
    for _ in 0..test_config.iterations {
        let _ = wgpu_tensor.clone().fft2(0, 1);
    }
        let wgpu_time = start.elapsed();
        
        println!("  ✅ 2D FFT: {:.2}ms avg", wgpu_time.as_millis() as f64 / test_config.iterations as f64);
        
        // Performance comparison
        let speedup = ndarray_time.as_nanos() as f64 / wgpu_time.as_nanos() as f64;
        println!("\n📊 Performance Comparison:");
        println!("  WGPU is {:.2}x {} than NdArray", 
                 if speedup > 1.0 { speedup } else { 1.0 / speedup },
                 if speedup > 1.0 { "faster" } else { "slower" });
        
        // Memory throughput calculation
        let bytes_per_op = test_config.size * test_config.size * 4 * 2; // Complex numbers
        let throughput_gb_s = (bytes_per_op as f64 * test_config.iterations as f64) / 
                             (wgpu_time.as_nanos() as f64 / 1_000_000_000.0) / 1_000_000_000.0;
        println!("  WGPU throughput: {:.2} GB/s", throughput_gb_s);
}
}

/// Correctness comparison between backends
pub fn verify_cross_backend_correctness() {
    println!("\n=== Cross-Backend Correctness Verification ===");
    
    let test_size = 32;
    let data: Vec<f32> = (0..test_size).map(|i| (i as f32).sin()).collect();
    
    // NdArray reference
    let ndarray_device = burn_ndarray::NdArrayDevice::Cpu;
    let ndarray_tensor = Tensor::<NdArrayBackend, 1>::from_floats(data.as_slice(), &ndarray_device);
    let ndarray_result = ndarray_tensor.cumsum(0);
    let ndarray_data = ndarray_result.into_data().to_vec::<f32>().unwrap();
    
    println!("🔄 NdArray cumsum computed ({} elements)", ndarray_data.len());
    
    // WGPU comparison (if available)
    let wgpu_device = WgpuDevice::default();
    let wgpu_tensor = Tensor::<WgpuBackend, 1>::from_floats(data.as_slice(), &wgpu_device);
        let wgpu_result = wgpu_tensor.cumsum(0);
        let wgpu_data = wgpu_result.into_data().to_vec::<f32>().unwrap();
        
        println!("🚀 WGPU cumsum computed ({} elements)", wgpu_data.len());
        
        // Compare results
        let mut max_diff = 0.0f32;
        let tolerance = 1e-5f32;
        let mut mismatches = 0;
        
        for (i, (&ndarray_val, &wgpu_val)) in ndarray_data.iter().zip(wgpu_data.iter()).enumerate() {
            let diff = (ndarray_val - wgpu_val).abs();
            max_diff = max_diff.max(diff);
            
            if diff > tolerance {
                if mismatches < 5 { // Only show first 5 mismatches
                    println!("  ⚠️  Mismatch at index {}: NdArray={:.6}, WGPU={:.6}, diff={:.2e}", 
                             i, ndarray_val, wgpu_val, diff);
                }
                mismatches += 1;
            }
        }
        
        if mismatches == 0 {
            println!("  ✅ Perfect match! Max difference: {:.2e}", max_diff);
        } else {
            println!("  ⚠️  {} mismatches found (tolerance: {:.2e})", mismatches, tolerance);
            println!("     Max difference: {:.2e}", max_diff);
        }
}
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scan_backends_available() {
        // Just verify that we can create tensors on both backends
        let data = vec![1.0, 2.0, 3.0, 4.0];
        
        // NdArray should always be available
        let ndarray_device = burn_ndarray::NdArrayDevice::Cpu;
        let _ndarray_tensor = Tensor::<NdArrayBackend, 1>::from_floats(data.as_slice(), &ndarray_device);
        
        // WGPU might not be available in all environments
        let wgpu_device = WgpuDevice::default();
        let _wgpu_tensor = Tensor::<WgpuBackend, 1>::from_floats(data.as_slice(), &wgpu_device);
    }
    
    #[test]
    fn test_cross_backend_scan_correctness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // NdArray reference
        let ndarray_device = burn_ndarray::NdArrayDevice::Cpu;
        let ndarray_tensor = Tensor::<NdArrayBackend, 1>::from_floats(data.as_slice(), &ndarray_device);
        let ndarray_result = ndarray_tensor.cumsum(0).into_data().to_vec::<f32>().unwrap();
        
        // Expected: [1.0, 3.0, 6.0, 10.0, 15.0]
        let expected = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        for (actual, expected) in ndarray_result.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6, 
                    "NdArray cumsum failed: got {}, expected {}", actual, expected);
        }
        
        // WGPU comparison (if available)
        let wgpu_device = WgpuDevice::default();
        let wgpu_tensor = Tensor::<WgpuBackend, 1>::from_floats(data.as_slice(), &wgpu_device);
            let wgpu_result = wgpu_tensor.cumsum(0).into_data().to_vec::<f32>().unwrap();
            
            for (wgpu_val, ndarray_val) in wgpu_result.iter().zip(ndarray_result.iter()) {
                assert!((wgpu_val - ndarray_val).abs() < 1e-6, 
                        "WGPU vs NdArray mismatch: {} vs {}", wgpu_val, ndarray_val);
            }
        }
    }
}

pub fn main() {
    println!("🔥 Burn Cross-Backend Testing Framework");
    println!("=====================================");
    
    // Run all benchmarks
    benchmark_scan_operations();
    benchmark_fft_operations();
    verify_cross_backend_correctness();
    
    println!("\n🎯 Testing completed!");
}
