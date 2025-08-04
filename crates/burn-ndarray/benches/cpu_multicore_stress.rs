#![recursion_limit = "256"]

use burn_ndarray::{NdArray, NdArrayDevice};
use burn_tensor::{Tensor, TensorData, Shape};
use std::time::Instant;
use criterion::{criterion_group, criterion_main, Criterion};
use rayon::prelude::*;

type CpuBackend = NdArray<f32>;

fn generate_data_parallel(size: usize) -> Vec<f32> {
    println!("🔥 Generating {} elements using {} CPU threads", size, rayon::current_num_threads());
    let start = Instant::now();
    
    let data = (0..size).into_par_iter()
        .map(|i| ((i % 7) + 1) as f32) // Use mod 7 for variety
        .collect();
    
    let duration = start.elapsed();
    let gb_per_sec = (size * 4) as f64 / duration.as_secs_f64() / 1e9;
    println!("✅ Data generation: {:?} ({:.2} GB/s)", duration, gb_per_sec);
    data
}

fn manual_multicore_scan(data: &[f32]) -> (Vec<f32>, std::time::Duration) {
    println!("🔥 Manual multicore scan using {} threads", rayon::current_num_threads());
    let start = Instant::now();
    
    let mut result = data.to_vec();
    let num_threads = rayon::current_num_threads();
    let chunk_size = (result.len() + num_threads - 1) / num_threads;
    
    // Phase 1: Parallel cumsum within chunks - ALL CORES WORKING
    let chunk_sums: Vec<f32> = result
        .par_chunks_mut(chunk_size)
        .map(|chunk| {
            for i in 1..chunk.len() {
                chunk[i] += chunk[i - 1];
            }
            chunk[chunk.len() - 1]
        })
        .collect();
    
    // Phase 2: Sequential prefix of chunk totals (small, fast)
    let mut prefix = 0.0;
    let mut chunk_prefixes = Vec::with_capacity(chunk_sums.len());
    for &sum in &chunk_sums {
        chunk_prefixes.push(prefix);
        prefix += sum;
    }
    
    // Phase 3: Add prefixes in parallel - ALL CORES WORKING AGAIN
    result.par_chunks_mut(chunk_size)
        .enumerate()
        .skip(1)
        .for_each(|(i, chunk)| {
            let prefix = chunk_prefixes[i];
            for elem in chunk {
                *elem += prefix;
            }
        });
    
    let duration = start.elapsed();
    println!("✅ Manual multicore scan: {:?}", duration);
    (result, duration)
}

fn bench_cpu_multicore_stress(c: &mut Criterion) {
    println!("\n🔥🔥🔥 CPU MULTICORE STRESS TEST 🔥🔥🔥");
    println!("=====================================");
    
    // Force all CPU threads to be used
    rayon::ThreadPoolBuilder::new()
        .num_threads(std::thread::available_parallelism().unwrap().get())
        .build_global()
        .unwrap();
    
    println!("🚀 Using {} CPU threads", rayon::current_num_threads());
    
    // Test with different sizes to really stress all cores
    let sizes = vec![
        ("10M", 10_000_000),
        ("50M", 50_000_000),
        ("100M", 100_000_000),
    ];
    
    for (name, size) in sizes {
        println!("\n🎯 Testing {} elements", name);
        
        // Generate data
        let data = generate_data_parallel(size);
        
        // Test manual multicore implementation
        let (manual_result, manual_time) = manual_multicore_scan(&data);
        println!("Manual result first 5: {:?}", &manual_result[0..5]);
        
        // Test burn's implementation
        let device = NdArrayDevice::default();
        let tensor: Tensor<CpuBackend, 1> = Tensor::from_data(
            TensorData::new(data.clone(), Shape::new([size])), &device
        );
        
        println!("🔥 Testing burn's cumsum with {} threads", rayon::current_num_threads());
        let burn_start = Instant::now();
        let burn_result = tensor.clone().cumsum(0);
        let burn_time = burn_start.elapsed();
        let burn_data: Vec<f32> = burn_result.to_data().to_vec().unwrap();
        println!("✅ Burn cumsum: {:?}", burn_time);
        println!("Burn result first 5: {:?}", &burn_data[0..5]);
        
        // Compare performance
        let speedup = burn_time.as_secs_f64() / manual_time.as_secs_f64();
        if manual_time < burn_time {
            println!("🏆 Manual multicore is {:.2}x FASTER than burn!", speedup);
        } else {
            println!("🏆 Burn is {:.2}x faster than manual multicore", 1.0/speedup);
        }
        
        // Verify correctness
        let matches = manual_result.iter().zip(burn_data.iter())
            .take(1000)
            .all(|(a, b)| (a - b).abs() < 1e-3);
        println!("✅ Results match: {}", matches);
        
        // Benchmark both approaches
        c.bench_function(&format!("manual_multicore_{}", name), |b| {
            b.iter(|| {
                manual_multicore_scan(&data);
            });
        });
        
        c.bench_function(&format!("burn_cumsum_{}", name), |b| {
            b.iter(|| {
                tensor.clone().cumsum(0)
            });
        });
    }
}

criterion_group!(benches, bench_cpu_multicore_stress);
criterion_main!(benches);
