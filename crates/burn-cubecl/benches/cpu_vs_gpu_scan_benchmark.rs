//! Comprehensive CPU vs GPU parallel scan benchmark
//! 
//! This benchmark compares:
//! - NdArray backend (CPU multi-core using Rayon)
//! - Wgpu backend (GPU parallel using compute shaders)
//! - Various tensor sizes and scan operations
//! - Performance metrics and utilization analysis

use burn_ndarray::NdArray;
use burn_wgpu::Wgpu;
use burn_tensor::{backend::Backend, Tensor, Distribution};
use std::time::Instant;

#[path = "benchmark_multicore_cpu_utilization.rs"]
mod benchmark_multicore_cpu_utilization;

fn main() {
    println!("🏁 Comprehensive CPU vs GPU Parallel Scan Benchmark");
    println!("🧵 CPU Backend: NdArray with Rayon ({} cores)", 
             std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    println!("🖥️  GPU Backend: Wgpu compute shaders");
    println!("📊 Comparing scan operations across multiple tensor sizes\n");
    
    benchmark_cpu_vs_gpu_performance();
    
    println!("\n🎯 CPU vs GPU benchmark completed!");
}

fn benchmark_cpu_vs_gpu_performance() {
    let cpu_device = burn_ndarray::NdArrayDevice::Cpu;
    let gpu_device = Default::default(); // Wgpu device
    
    let test_sizes = vec![
        ("1K", 1_000),
        ("10K", 10_000),
        ("100K", 100_000),
        ("1M", 1_000_000),
        ("5M", 5_000_000),
    ];
    
    println!("┌─────────┬─────────────────────────┬─────────────────────────┬──────────────┐");
    println!("│  Size   │        CPU (NdArray)    │       GPU (Wgpu)        │   Speedup    │");
    println!("├─────────┼─────────────────────────┼─────────────────────────┼──────────────┤");
    
    for (name, size) in test_sizes {
        // CPU benchmark
        let cpu_tensor: Tensor<NdArray<f32>, 1> = Tensor::random(
            [size], Distribution::Uniform(0.0, 1.0), &cpu_device
        );
        
        let cpu_start = Instant::now();
        let _cpu_result = cpu_tensor.cumsum(0);
        let cpu_time = cpu_start.elapsed();
        
        // GPU benchmark  
        let gpu_tensor: Tensor<Wgpu, 1> = Tensor::random(
            [size], Distribution::Uniform(0.0, 1.0), &gpu_device
        );
        
        let gpu_start = Instant::now();
        let _gpu_result = gpu_tensor.cumsum(0);
        let gpu_time = gpu_start.elapsed();
        
        let cpu_throughput = size as f64 / cpu_time.as_secs_f64() / 1_000_000.0;
        let gpu_throughput = size as f64 / gpu_time.as_secs_f64() / 1_000_000.0;
        let speedup = gpu_throughput / cpu_throughput;
        
        println!("│ {:>7} │ {:>8.2}ms {:>6.1}M/s │ {:>8.2}ms {:>6.1}M/s │ {:>10.2}x │",
                 name,
                 cpu_time.as_secs_f64() * 1000.0, cpu_throughput,
                 gpu_time.as_secs_f64() * 1000.0, gpu_throughput,
                 speedup);
    }
    
    println!("└─────────┴─────────────────────────┴─────────────────────────┴──────────────┘");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_gpu_correctness_comparison() {
        let cpu_device = burn_ndarray::NdArrayDevice::Cpu;
        let gpu_device = Default::default();
        
        // Test with known data to verify both backends produce same results
        let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let expected_cumsum = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        
        let cpu_tensor: Tensor<NdArray<f32>, 1> = Tensor::from_data(
            burn_tensor::TensorData::new(test_data.clone(), burn_tensor::Shape::new([5])), 
            &cpu_device
        );
        
        let gpu_tensor: Tensor<Wgpu, 1> = Tensor::from_data(
            burn_tensor::TensorData::new(test_data, burn_tensor::Shape::new([5])), 
            &gpu_device
        );
        
        let cpu_result = cpu_tensor.cumsum(0);
        let gpu_result = gpu_tensor.cumsum(0);
        
        let cpu_values: Vec<f32> = cpu_result.to_data().to_vec().unwrap();
        let gpu_values: Vec<f32> = gpu_result.to_data().to_vec().unwrap();
        
        for i in 0..5 {
            assert!((cpu_values[i] - expected_cumsum[i]).abs() < 1e-6,
                    "CPU result mismatch at {}: expected {}, got {}", i, expected_cumsum[i], cpu_values[i]);
            assert!((gpu_values[i] - expected_cumsum[i]).abs() < 1e-6,
                    "GPU result mismatch at {}: expected {}, got {}", i, expected_cumsum[i], gpu_values[i]);
            assert!((cpu_values[i] - gpu_values[i]).abs() < 1e-6,
                    "CPU/GPU mismatch at {}: CPU={}, GPU={}", i, cpu_values[i], gpu_values[i]);
        }
        
        println!("✅ CPU vs GPU correctness verified - both produce identical results");
    }
}
