#![recursion_limit = "256"]

use burn_ndarray::{NdArray, NdArrayDevice};
use burn_wgpu::{Wgpu, WgpuDevice};
use burn_tensor::{Tensor, TensorData, Shape, Device as _};
use std::time::Instant;
use criterion::{criterion_group, criterion_main, Criterion};
use rayon::prelude::*;

type CpuBackend = NdArray<f32>;
type GpuBackend = Wgpu<f32>;

fn generate_data(size: usize) -> Vec<f32> {
    (0..size).into_par_iter()
        .map(|i| ((i % 5) + 1) as f32)
        .collect()
}

fn bench_cpu_vs_gpu_scan(c: &mut Criterion) {
    println!("🔥 CPU vs GPU Scan Comparison - 100M elements");
    
    // Use 100M elements instead of 1G to avoid GPU memory issues
    let size = 100_000_000;
    
    // Setup data
    let data = generate_data(size);
    println!("✅ Generated {}M elements ({:.1} GB)", size / 1_000_000, (size * 4) as f64 / 1e9);
    
    // Setup CPU
    let cpu_device = NdArrayDevice::default();
    let cpu_tensor: Tensor<CpuBackend, 1> = Tensor::from_data(
        TensorData::new(data.clone(), Shape::new([size])), &cpu_device
    );
    println!("✅ CPU tensor ready");
    
    // Setup GPU
    let gpu_device = WgpuDevice::default();
    let gpu_tensor: Tensor<GpuBackend, 1> = Tensor::from_data(
        TensorData::new(data.clone(), Shape::new([size])), &gpu_device
    );
    println!("✅ GPU tensor ready");
    
    // Direct comparison - single run
    println!("\n🏁 DIRECT COMPARISON:");
    
    // CPU run
    let cpu_start = Instant::now();
    let _cpu_result = cpu_tensor.clone().cumsum(0);
    let cpu_time = cpu_start.elapsed();
    println!("🖥️  CPU (multi-core): {:?}", cpu_time);
    
    // GPU run
    let gpu_start = Instant::now();
    let _gpu_result = gpu_tensor.clone().cumsum(0);
    let gpu_time = gpu_start.elapsed();
    println!("🎮 GPU: {:?}", gpu_time);
    
    // Winner
    if cpu_time < gpu_time {
        let speedup = gpu_time.as_secs_f64() / cpu_time.as_secs_f64();
        println!("� CPU WINS! {:.2}x faster than GPU", speedup);
    } else {
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        println!("🏆 GPU WINS! {:.2}x faster than CPU", speedup);
    }
    
    // Benchmark CPU (multi-core)
    c.bench_function("cpu_multicore_scan_100m", |b| {
        b.iter(|| {
            let start = Instant::now();
            let result = cpu_tensor.clone().cumsum(0);
            let duration = start.elapsed();
            println!("🖥️  CPU (multi-core): {:?}", duration);
            result
        });
    });
    
    // Benchmark GPU
    c.bench_function("gpu_scan_100m", |b| {
        b.iter(|| {
            let start = Instant::now();
            let result = gpu_tensor.clone().cumsum(0);
            let duration = start.elapsed();
            println!("� GPU: {:?}", duration);
            result
        });
    });
}

criterion_group!(benches, bench_cpu_vs_gpu_scan);
criterion_main!(benches);
