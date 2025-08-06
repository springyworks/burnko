#![recursion_limit = "256"]

use burn_ndarray::{NdArray, NdArrayDevice};
use burn_wgpu::{Wgpu, WgpuDevice};
use burn_tensor::{Tensor, TensorData, Shape};
use std::time::Instant;
use criterion::{criterion_group, criterion_main, Criterion};
use rayon::prelude::*;

type CpuBackend = NdArray<f32>;
type GpuBackend = Wgpu<f32>;

fn generate_data_parallel(size: usize) -> Vec<f32> {
    (0..size).into_par_iter()
        .map(|i| ((i % 7) + 1) as f32)
        .collect()
}

fn bench_cpu_vs_gpu_comparison(c: &mut Criterion) {
    println!("🔥🔥🔥 CPU MULTICORE vs GPU COMPARISON 🔥🔥🔥");
    println!("===================================================");
    
    // Test sizes that fit in GPU memory
    let sizes = vec![
        ("1M", 1_000_000),
        ("10M", 10_000_000),
        ("50M", 50_000_000),
    ];
    
    for (name, size) in sizes {
        println!("\n🎯 Testing {} elements ({:.1} MB)", name, (size * 4) as f64 / 1e6);
        
        // Generate data
        let data_start = Instant::now();
        let data = generate_data_parallel(size);
        let data_time = data_start.elapsed();
        println!("✅ Data generation: {:?}", data_time);
        
        // Setup CPU tensor
        let cpu_device = NdArrayDevice::default();
        let cpu_tensor: Tensor<CpuBackend, 1> = Tensor::from_data(
            TensorData::new(data.clone(), Shape::new([size])), &cpu_device
        );
        
        // Setup GPU tensor
        let gpu_device = WgpuDevice::default();
        let gpu_tensor: Tensor<GpuBackend, 1> = Tensor::from_data(
            TensorData::new(data.clone(), Shape::new([size])), &gpu_device
        );
        
        println!("✅ Both tensors ready");
        
        // 🖥️ CPU MULTICORE TEST
        println!("\n🖥️  CPU MULTICORE TEST (expect 70% across all cores):");
        let mut cpu_times = Vec::new();
        for i in 1..=5 {
            let cpu_start = Instant::now();
            let _cpu_result = cpu_tensor.clone().cumsum(0);
            let cpu_time = cpu_start.elapsed();
            cpu_times.push(cpu_time);
            println!("  Run {}: {:?}", i, cpu_time);
        }
        let cpu_avg = cpu_times.iter().sum::<std::time::Duration>() / cpu_times.len() as u32;
        println!("  🏆 CPU Average: {:?}", cpu_avg);
        
        // 🎮 GPU FULL-BURN TEST  
        println!("\n🎮 GPU FULL-BURN TEST (expect GPU load spike):");
        let mut gpu_times = Vec::new();
        for i in 1..=5 {
            let gpu_start = Instant::now();
            let _gpu_result = gpu_tensor.clone().cumsum(0);
            let gpu_time = gpu_start.elapsed();
            gpu_times.push(gpu_time);
            println!("  Run {}: {:?}", i, gpu_time);
        }
        let gpu_avg = gpu_times.iter().sum::<std::time::Duration>() / gpu_times.len() as u32;
        println!("  🏆 GPU Average: {:?}", gpu_avg);
        
        // 🏁 THE VERDICT
        println!("\n🏁 THE VERDICT FOR {}:", name);
        if cpu_avg < gpu_avg {
            let speedup = gpu_avg.as_secs_f64() / cpu_avg.as_secs_f64();
            println!("  🏆🏆🏆 CPU MULTICORE WINS! {:.2}x FASTER than GPU! 🏆🏆🏆", speedup);
            
            let cpu_gb_s = (size * 4) as f64 / cpu_avg.as_secs_f64() / 1e9;
            let gpu_gb_s = (size * 4) as f64 / gpu_avg.as_secs_f64() / 1e9;
            println!("  📊 CPU Throughput: {:.2} GB/s", cpu_gb_s);
            println!("  📊 GPU Throughput: {:.2} GB/s", gpu_gb_s);
        } else {
            let speedup = cpu_avg.as_secs_f64() / gpu_avg.as_secs_f64();
            println!("  🏆🏆🏆 GPU WINS! {:.2}x FASTER than CPU multicore! 🏆🏆🏆", speedup);
            
            let cpu_gb_s = (size * 4) as f64 / cpu_avg.as_secs_f64() / 1e9;
            let gpu_gb_s = (size * 4) as f64 / gpu_avg.as_secs_f64() / 1e9;
            println!("  📊 CPU Throughput: {:.2} GB/s", cpu_gb_s);
            println!("  📊 GPU Throughput: {:.2} GB/s", gpu_gb_s);
        }
        
        // Criterion benchmarks for official results
        c.bench_function(&format!("cpu_multicore_{}", name), |b| {
            b.iter(|| {
                cpu_tensor.clone().cumsum(0)
            });
        });
        
        c.bench_function(&format!("gpu_fullburn_{}", name), |b| {
            b.iter(|| {
                gpu_tensor.clone().cumsum(0)
            });
        });
        
        println!("{}", "=".repeat(60));
    }
    
    println!("\n🎉 COMPARISON COMPLETE! 🎉");
    println!("Watch your system monitor to see:");
    println!("  • CPU cores at 70% during CPU tests");
    println!("  • GPU load spikes during GPU tests");
}

criterion_group!(benches, bench_cpu_vs_gpu_comparison);
criterion_main!(benches);
