#![recursion_limit = "256"]

use burn_tensor::{Tensor, TensorData, Shape};
use burn_ndarray::NdArray;
use burn_wgpu::Wgpu;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Instant;

type CpuBackend = NdArray<f32>;
type GpuBackend = Wgpu<f32, i32>;

/// Quick CPU vs GPU comparison - small sizes for fast results
fn quick_cpu_vs_gpu_comparison(c: &mut Criterion) {
    let cpu_device = Default::default();
    let gpu_device = Default::default();
    
    println!("🚀 Quick CPU vs GPU Comparison");
    
    // Small, fast test sizes
    let test_sizes = vec![1_000, 10_000, 100_000];
    
    for &size in &test_sizes {
        println!("Testing size: {} elements", size);
        
        // Sequential data for predictable results
        let data: Vec<f32> = (1..=size).map(|x| x as f32).collect();
        
        // CPU test
        let start = Instant::now();
        let tensor_cpu: Tensor<CpuBackend, 1> = Tensor::from_data(
            TensorData::new(data.clone(), Shape::new([size])), &cpu_device
        );
        let result_cpu = tensor_cpu.cumsum(0);
        let _cpu_data: Vec<f32> = result_cpu.to_data().to_vec().unwrap();
        let cpu_duration = start.elapsed();
        
        // GPU test
        let start = Instant::now();
        let tensor_gpu: Tensor<GpuBackend, 1> = Tensor::from_data(
            TensorData::new(data.clone(), Shape::new([size])), &gpu_device
        );
        let result_gpu = tensor_gpu.cumsum(0);
        let _gpu_data: Vec<f32> = result_gpu.to_data().to_vec().unwrap();
        let gpu_duration = start.elapsed();
        
        println!("  CPU: {:?} ({:.2} Melems/sec)", 
                 cpu_duration, size as f64 / cpu_duration.as_secs_f64() / 1_000_000.0);
        println!("  GPU: {:?} ({:.2} Melems/sec)", 
                 gpu_duration, size as f64 / gpu_duration.as_secs_f64() / 1_000_000.0);
        
        let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
        if speedup > 1.0 {
            println!("  🏆 GPU is {:.2}x faster", speedup);
        } else {
            println!("  🏆 CPU is {:.2}x faster", 1.0 / speedup);
        }
        println!();
    }
    
    // Simple criterion benchmark for smallest size
    let size = 10_000;
    let data: Vec<f32> = (1..=size).map(|x| x as f32).collect();
    
    c.bench_function("quick_cpu_10k", |b| {
        let tensor_cpu: Tensor<CpuBackend, 1> = Tensor::from_data(
            TensorData::new(data.clone(), Shape::new([size])), &cpu_device
        );
        b.iter(|| {
            let result = black_box(tensor_cpu.clone().cumsum(0));
            black_box(result)
        });
    });
    
    c.bench_function("quick_gpu_10k", |b| {
        let tensor_gpu: Tensor<GpuBackend, 1> = Tensor::from_data(
            TensorData::new(data.clone(), Shape::new([size])), &gpu_device
        );
        b.iter(|| {
            let result = black_box(tensor_gpu.clone().cumsum(0));
            black_box(result)
        });
    });
}

/// Quick verification that CPU and GPU give same results
fn quick_verification_test(c: &mut Criterion) {
    let cpu_device = Default::default();
    let gpu_device = Default::default();
    
    println!("🔍 Quick Verification Test");
    
    // Test data: [1, 2, 3, 4, 5] -> expected [1, 3, 6, 10, 15]
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let expected = vec![1.0, 3.0, 6.0, 10.0, 15.0];
    
    c.bench_function("verification", |b| {
        b.iter(|| {
            // CPU result
            let tensor_cpu: Tensor<CpuBackend, 1> = Tensor::from_data(
                TensorData::new(data.clone(), Shape::new([5])), &cpu_device
            );
            let result_cpu = tensor_cpu.cumsum(0);
            let cpu_values: Vec<f32> = result_cpu.to_data().to_vec().unwrap();
            
            // GPU result
            let tensor_gpu: Tensor<GpuBackend, 1> = Tensor::from_data(
                TensorData::new(data.clone(), Shape::new([5])), &gpu_device
            );
            let result_gpu = tensor_gpu.cumsum(0);
            let gpu_values: Vec<f32> = result_gpu.to_data().to_vec().unwrap();
            
            // Verify both match expected
            for (i, ((cpu, gpu), expected)) in cpu_values.iter()
                .zip(gpu_values.iter())
                .zip(expected.iter())
                .enumerate() {
                assert!((cpu - expected).abs() < 1e-6, 
                    "CPU mismatch at {}: expected {}, got {}", i, expected, cpu);
                assert!((gpu - expected).abs() < 1e-6, 
                    "GPU mismatch at {}: expected {}, got {}", i, expected, gpu);
                assert!((cpu - gpu).abs() < 1e-6,
                    "CPU/GPU mismatch at {}: CPU={}, GPU={}", i, cpu, gpu);
            }
            
            black_box((cpu_values, gpu_values))
        });
    });
    
    println!("✅ Verification passed - CPU and GPU give identical results");
}

criterion_group!(
    name = quick_benches;
    config = Criterion::default().sample_size(20).measurement_time(std::time::Duration::from_secs(5));
    targets = quick_verification_test, quick_cpu_vs_gpu_comparison
);
criterion_main!(quick_benches);
