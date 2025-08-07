//! Cross-Backend Performance Comparison Tests
//! 
//! This test compares NdArray (CPU) vs WGPU (GPU) performance for scan and FFT operations
//! These tests span multiple crates and provide comprehensive benchmarking.

use burn_tensor::{Tensor, TensorData, Shape, Distribution, ops::{ScanConfig, ScanOp}};
use burn_ndarray::NdArray;
use burn_wgpu::Wgpu;
use std::time::{Duration, Instant};

type CpuBackend = NdArray<f32>;
type GpuBackend = Wgpu<f32>;

/// Comprehensive comparison test between CPU and GPU backends
#[test]
fn test_cpu_vs_gpu_scan_performance() {
    println!("\n🏁 CPU vs GPU SCAN PERFORMANCE COMPARISON 🏁");
    
    let sizes = vec![
        (128, 128),   // 16K elements
        (256, 256),   // 64K elements
        (512, 512),   // 256K elements
        (1024, 1024), // 1M elements
    ];
    
    for (rows, cols) in sizes {
        println!("\n📊 Testing {}×{} matrix ({} elements)...", rows, cols, rows * cols);
        
        let values: Vec<f32> = (0..rows*cols)
            .map(|i| (i as f32 * 0.001) % 10.0 + 1.0)
            .collect();
        
        // CPU Test
        let cpu_device = Default::default();
        let cpu_tensor = Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(values.clone(), Shape::new([rows, cols])), 
            &cpu_device
        );
        
        let start_time = Instant::now();
        let cpu_config = ScanConfig::new(ScanOp::Add, 0);
        let cpu_result = cpu_tensor.scan(cpu_config);
        let _cpu_values: Vec<f32> = cpu_result.to_data().to_vec().unwrap();
        let cpu_time = start_time.elapsed();
        
        // GPU Test
        let gpu_device = Default::default();
        let gpu_tensor = Tensor::<GpuBackend, 2>::from_data(
            TensorData::new(values, Shape::new([rows, cols])), 
            &gpu_device
        );
        
        let start_time = Instant::now();
        let gpu_config = ScanConfig::new(ScanOp::Add, 0);
        let gpu_result = gpu_tensor.scan(gpu_config);
        let _gpu_values: Vec<f32> = gpu_result.to_data().to_vec().unwrap();
        let gpu_time = start_time.elapsed();
        
        // Performance comparison
        let cpu_throughput = (rows * cols) as f64 / cpu_time.as_secs_f64() / 1_000_000.0;
        let gpu_throughput = (rows * cols) as f64 / gpu_time.as_secs_f64() / 1_000_000.0;
        let speedup = gpu_throughput / cpu_throughput;
        
        println!("   CPU: {:?} ({:.2} Melems/sec)", cpu_time, cpu_throughput);
        println!("   GPU: {:?} ({:.2} Melems/sec)", gpu_time, gpu_throughput);
        
        if speedup > 1.0 {
            println!("   🚀 GPU is {:.2}x faster!", speedup);
        } else {
            println!("   🏃 CPU is {:.2}x faster", 1.0 / speedup);
        }
    }
}

#[test]
fn test_cpu_vs_gpu_fft_performance() {
    println!("\n🌊 CPU vs GPU FFT PERFORMANCE COMPARISON 🌊");
    
    let sizes = vec![64, 128, 256, 512];
    
    for size in sizes {
        println!("\n📊 Testing {}×{} FFT ({} elements)...", size, size, size * size);
        
        // CPU Test
        let cpu_device = Default::default();
        let cpu_tensor: Tensor<CpuBackend, 2> = Tensor::random(
            [size, size],
            Distribution::Uniform(-1.0, 1.0),
            &cpu_device,
        );
        
        let start_time = Instant::now();
        let cpu_fft = cpu_tensor.fft(0);
        let _cpu_ifft = cpu_fft.ifft(0);
        let cpu_time = start_time.elapsed();
        
        // GPU Test
        let gpu_device = Default::default();
        let gpu_tensor: Tensor<GpuBackend, 2> = Tensor::random(
            [size, size],
            Distribution::Uniform(-1.0, 1.0),
            &gpu_device,
        );
        
        let start_time = Instant::now();
        let gpu_fft = gpu_tensor.fft(0);
        let _gpu_ifft = gpu_fft.ifft(0);
        let gpu_time = start_time.elapsed();
        
        // Performance comparison
        let cpu_throughput = (size * size) as f64 / cpu_time.as_secs_f64() / 1_000_000.0;
        let gpu_throughput = (size * size) as f64 / gpu_time.as_secs_f64() / 1_000_000.0;
        let speedup = gpu_throughput / cpu_throughput;
        
        println!("   CPU: {:?} ({:.2} Melems/sec)", cpu_time, cpu_throughput);
        println!("   GPU: {:?} ({:.2} Melems/sec)", gpu_time, gpu_throughput);
        
        if speedup > 1.0 {
            println!("   🚀 GPU is {:.2}x faster!", speedup);
        } else {
            println!("   🏃 CPU is {:.2}x faster", 1.0 / speedup);
        }
    }
}
