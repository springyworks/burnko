#![recursion_limit = "256"]

use burn_ndarray::{NdArray, NdArrayDevice};
use burn_tensor::{Tensor, TensorData, Shape};
use std::time::Instant;
use criterion::{criterion_group, criterion_main, Criterion};
use rayon::prelude::*;

type CpuBackend = NdArray<f32>;

fn generate_data(size: usize) -> Vec<f32> {
    // Just generate the data fast with rayon
    (0..size).into_par_iter()
        .map(|i| ((i % 5) + 1) as f32)
        .collect()
}

fn manual_parallel_cumsum(data: &mut [f32]) -> std::time::Duration {
    let start = Instant::now();
    
    // Parallel chunked cumsum to force CPU utilization
    let num_threads = rayon::current_num_threads();
    let chunk_size = data.len() / num_threads;
    
    if chunk_size > 0 {
        // Phase 1: Parallel cumsum within chunks
        let chunk_sums: Vec<f32> = data
            .par_chunks_mut(chunk_size)
            .map(|chunk| {
                for i in 1..chunk.len() {
                    chunk[i] += chunk[i - 1];
                }
                chunk[chunk.len() - 1]
            })
            .collect();
        
        // Phase 2: Sequential prefix sum of chunk totals
        let mut prefix = 0.0;
        let mut chunk_prefixes = Vec::with_capacity(chunk_sums.len());
        for &sum in &chunk_sums {
            chunk_prefixes.push(prefix);
            prefix += sum;
        }
        
        // Phase 3: Add prefix to each chunk in parallel
        data.par_chunks_mut(chunk_size)
            .enumerate()
            .skip(1)
            .for_each(|(i, chunk)| {
                let prefix = chunk_prefixes[i];
                for elem in chunk {
                    *elem += prefix;
                }
            });
    }
    
    start.elapsed()
}

fn bench_cpu_scan_1g(c: &mut Criterion) {
    println!("🔥 CPU Scan 1G elements");
    
    // Force rayon to use all threads
    rayon::ThreadPoolBuilder::new()
        .num_threads(std::thread::available_parallelism().unwrap().get())
        .build_global()
        .unwrap();
    
    println!("🚀 Using {} CPU threads", rayon::current_num_threads());
    
    let data = generate_data(1_000_000_000);
    let device = NdArrayDevice::default();
    let tensor: Tensor<CpuBackend, 1> = Tensor::from_data(
        TensorData::new(data.clone(), Shape::new([1_000_000_000])), &device
    );
    
    // Test manual parallel cumsum first
    let mut manual_data = data.clone();
    let manual_time = manual_parallel_cumsum(&mut manual_data);
    println!("🔥 Manual parallel cumsum: {:?}", manual_time);
    
    c.bench_function("cpu_scan_1g", |b| {
        b.iter(|| {
            let start = Instant::now();
            let result = tensor.clone().cumsum(0);
            let duration = start.elapsed();
            println!("CPU scan (burn): {:?}", duration);
            result
        });
    });
}

criterion_group!(benches, bench_cpu_scan_1g);
criterion_main!(benches);
