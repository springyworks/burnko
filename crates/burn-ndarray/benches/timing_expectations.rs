#![recursion_limit = "256"]

use burn_ndarray::{NdArray, NdArrayDevice};
use burn_tensor::{Tensor, TensorData, Shape};
use std::time::Instant;
use criterion::{criterion_group, criterion_main, Criterion};
use rayon::prelude::*;

type CpuBackend = NdArray<f32>;

fn quick_timing_test() {
    println!("🔥 QUICK CPU MULTICORE TIMING TEST");
    println!("=================================");
    println!("Using {} CPU threads", rayon::current_num_threads());
    
    let sizes = vec![
        ("1M", 1_000_000, "~10ms"),
        ("10M", 10_000_000, "~100ms"), 
        ("50M", 50_000_000, "~500ms"),
        ("100M", 100_000_000, "~1s"),
    ];
    
    for (name, size, expected) in sizes {
        println!("\n🎯 Testing {} elements (expected: {})", name, expected);
        
        // Generate data
        let data_start = Instant::now();
        let data: Vec<f32> = (0..size).into_par_iter()
            .map(|i| (i % 7) as f32 + 1.0)
            .collect();
        let data_time = data_start.elapsed();
        println!("  Data generation: {:?}", data_time);
        
        // Create tensor
        let tensor_start = Instant::now();
        let device = NdArrayDevice::default();
        let tensor: Tensor<CpuBackend, 1> = Tensor::from_data(
            TensorData::new(data, Shape::new([size])), &device
        );
        let tensor_time = tensor_start.elapsed();
        println!("  Tensor creation: {:?}", tensor_time);
        
        // Cumsum operation
        let scan_start = Instant::now();
        let _result = tensor.cumsum(0);
        let scan_time = scan_start.elapsed();
        println!("  ✅ Cumsum operation: {:?}", scan_time);
        
        let total = data_time + tensor_time + scan_time;
        println!("  📊 Total time: {:?}", total);
        
        // Performance metrics
        let gb_per_sec = (size * 4) as f64 / scan_time.as_secs_f64() / 1e9;
        println!("  📈 Throughput: {:.1} GB/s", gb_per_sec);
    }
}

fn bench_timing_expectations(c: &mut Criterion) {
    // Run the quick test first
    quick_timing_test();
    
    println!("\n🏁 BENCHMARK TIMING EXPECTATIONS:");
    println!("- Small (1M):   each iteration ~10ms,  full benchmark ~30s");
    println!("- Medium (10M): each iteration ~100ms, full benchmark ~5min");
    println!("- Large (100M): each iteration ~1s,    full benchmark ~20min");
    println!("\nIf it takes longer than expected, it might be hanging!");
    
    // Quick benchmark - just 1M elements with few samples
    let data: Vec<f32> = (0..1_000_000).into_par_iter()
        .map(|i| (i % 7) as f32 + 1.0)
        .collect();
    
    let device = NdArrayDevice::default();
    let tensor: Tensor<CpuBackend, 1> = Tensor::from_data(
        TensorData::new(data, Shape::new([1_000_000])), &device
    );
    
    let mut group = c.benchmark_group("timing_test");
    group.sample_size(10);  // Small sample size for quick results
    
    group.bench_function("cpu_cumsum_1M", |b| {
        b.iter(|| {
            let start = Instant::now();
            let result = tensor.clone().cumsum(0);
            let duration = start.elapsed();
            println!("    Iteration time: {:?}", duration);
            result
        });
    });
    
    group.finish();
    println!("\n✅ Quick benchmark complete! This should have taken ~30 seconds total.");
}

criterion_group!(benches, bench_timing_expectations);
criterion_main!(benches);
