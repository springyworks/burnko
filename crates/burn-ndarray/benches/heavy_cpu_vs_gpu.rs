#![recursion_limit = "256"]

use burn_tensor::{Tensor, TensorData, Shape};
use burn_ndarray::NdArray;
use burn_wgpu::Wgpu;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Instant;

type CpuBackend = NdArray<f32>;
type GpuBackend = Wgpu<f32, i32>;

/// Heavy lifting test - find the crossover point where GPU might win
fn heavy_cpu_vs_gpu_battle(c: &mut Criterion) {
    let cpu_device = Default::default();
    let gpu_device = Default::default();
    
    println!("💀 HEAVY CPU vs GPU BATTLE - Finding the crossover point");
    
    // Large sizes where GPU might start winning
    let large_sizes = vec![500_000, 1_000_000, 2_000_000, 5_000_000];
    
    let mut group = c.benchmark_group("Heavy CPU vs GPU Battle");
    group.sample_size(10).measurement_time(std::time::Duration::from_secs(10));
    
    for &size in &large_sizes {
        println!("🔥 Heavy battle size: {} elements ({:.1} MB)", size, (size * 4) as f64 / 1_000_000.0);
        
        // Sequential data
        let data: Vec<f32> = (1..=size).map(|x| (x % 10000) as f32).collect(); // Modulo to prevent overflow
        
        // Manual timing first
        let start = Instant::now();
        let tensor_cpu: Tensor<CpuBackend, 1> = Tensor::from_data(
            TensorData::new(data.clone(), Shape::new([size])), &cpu_device
        );
        let result_cpu = tensor_cpu.cumsum(0);
        let _cpu_data: Vec<f32> = result_cpu.to_data().to_vec().unwrap();
        let cpu_duration = start.elapsed();
        
        let start = Instant::now();
        let tensor_gpu: Tensor<GpuBackend, 1> = Tensor::from_data(
            TensorData::new(data.clone(), Shape::new([size])), &gpu_device
        );
        let result_gpu = tensor_gpu.cumsum(0);
        let _gpu_data: Vec<f32> = result_gpu.to_data().to_vec().unwrap();
        let gpu_duration = start.elapsed();
        
        let cpu_throughput = size as f64 / cpu_duration.as_secs_f64() / 1_000_000.0;
        let gpu_throughput = size as f64 / gpu_duration.as_secs_f64() / 1_000_000.0;
        
        println!("  CPU: {:?} ({:.2} Melems/sec)", cpu_duration, cpu_throughput);
        println!("  GPU: {:?} ({:.2} Melems/sec)", gpu_duration, gpu_throughput);
        
        let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
        if speedup > 1.0 {
            println!("  🏆 GPU is {:.2}x faster", speedup);
        } else {
            println!("  🏆 CPU is {:.2}x faster", 1.0 / speedup);
        }
        println!();
        
        // Criterion benchmarks for smaller sizes only (to avoid taking forever)
        if size <= 2_000_000 {
            group.bench_with_input(
                BenchmarkId::new("Heavy_CPU", size),
                &size,
                |b, _| {
                    let tensor_cpu: Tensor<CpuBackend, 1> = Tensor::from_data(
                        TensorData::new(data.clone(), Shape::new([size])), &cpu_device
                    );
                    b.iter(|| {
                        let result = black_box(tensor_cpu.clone().cumsum(0));
                        black_box(result)
                    });
                },
            );
            
            group.bench_with_input(
                BenchmarkId::new("Heavy_GPU", size),
                &size,
                |b, _| {
                    let tensor_gpu: Tensor<GpuBackend, 1> = Tensor::from_data(
                        TensorData::new(data.clone(), Shape::new([size])), &gpu_device
                    );
                    b.iter(|| {
                        let result = black_box(tensor_gpu.clone().cumsum(0));
                        black_box(result)
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Test 2D and 3D tensors to see if GPU wins on multi-dimensional data
fn multidimensional_heavy_battle(_c: &mut Criterion) {
    let cpu_device = Default::default();
    let gpu_device = Default::default();
    
    println!("🎯 MULTIDIMENSIONAL HEAVY BATTLE - 2D and 3D tensors");
    
    // 2D tensor tests
    let tensor_2d_configs = vec![
        (1000, 1000),   // 1M elements
        (1414, 1414),   // ~2M elements
        (2236, 2236),   // ~5M elements
    ];
    
    for &(rows, cols) in &tensor_2d_configs {
        let total_elements = rows * cols;
        println!("🔥 2D Battle: {}x{} = {} elements", rows, cols, total_elements);
        
        let data: Vec<f32> = (1..=total_elements).map(|x| (x % 1000) as f32).collect();
        
        // Test both axes
        for axis in 0..2 {
            println!("  Testing axis {}", axis);
            
            let start = Instant::now();
            let tensor_cpu: Tensor<CpuBackend, 2> = Tensor::from_data(
                TensorData::new(data.clone(), Shape::new([rows, cols])), &cpu_device
            );
            let result_cpu = tensor_cpu.cumsum(axis);
            let _cpu_data: Vec<f32> = result_cpu.to_data().to_vec().unwrap();
            let cpu_duration = start.elapsed();
            
            let start = Instant::now();
            let tensor_gpu: Tensor<GpuBackend, 2> = Tensor::from_data(
                TensorData::new(data.clone(), Shape::new([rows, cols])), &gpu_device
            );
            let result_gpu = tensor_gpu.cumsum(axis);
            let _gpu_data: Vec<f32> = result_gpu.to_data().to_vec().unwrap();
            let gpu_duration = start.elapsed();
            
            let cpu_throughput = total_elements as f64 / cpu_duration.as_secs_f64() / 1_000_000.0;
            let gpu_throughput = total_elements as f64 / gpu_duration.as_secs_f64() / 1_000_000.0;
            
            println!("    CPU: {:?} ({:.2} Melems/sec)", cpu_duration, cpu_throughput);
            println!("    GPU: {:?} ({:.2} Melems/sec)", gpu_duration, gpu_throughput);
            
            let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
            if speedup > 1.0 {
                println!("    🏆 GPU is {:.2}x faster", speedup);
            } else {
                println!("    🏆 CPU is {:.2}x faster", 1.0 / speedup);
            }
        }
        println!();
    }
    
    // 3D tensor test - one quick test
    let (depth, height, width) = (100, 100, 100); // 1M elements
    let total_elements = depth * height * width;
    println!("🔥 3D Battle: {}x{}x{} = {} elements", depth, height, width, total_elements);
    
    let data: Vec<f32> = (1..=total_elements).map(|x| (x % 1000) as f32).collect();
    
    for axis in 0..3 {
        println!("  Testing axis {}", axis);
        
        let start = Instant::now();
        let tensor_cpu: Tensor<CpuBackend, 3> = Tensor::from_data(
            TensorData::new(data.clone(), Shape::new([depth, height, width])), &cpu_device
        );
        let result_cpu = tensor_cpu.cumsum(axis);
        let _cpu_data: Vec<f32> = result_cpu.to_data().to_vec().unwrap();
        let cpu_duration = start.elapsed();
        
        let start = Instant::now();
        let tensor_gpu: Tensor<GpuBackend, 3> = Tensor::from_data(
            TensorData::new(data.clone(), Shape::new([depth, height, width])), &gpu_device
        );
        let result_gpu = tensor_gpu.cumsum(axis);
        let _gpu_data: Vec<f32> = result_gpu.to_data().to_vec().unwrap();
        let gpu_duration = start.elapsed();
        
        let cpu_throughput = total_elements as f64 / cpu_duration.as_secs_f64() / 1_000_000.0;
        let gpu_throughput = total_elements as f64 / gpu_duration.as_secs_f64() / 1_000_000.0;
        
        println!("    CPU: {:?} ({:.2} Melems/sec)", cpu_duration, cpu_throughput);
        println!("    GPU: {:?} ({:.2} Melems/sec)", gpu_duration, gpu_throughput);
        
        let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
        if speedup > 1.0 {
            println!("    🏆 GPU is {:.2}x faster", speedup);
        } else {
            println!("    🏆 CPU is {:.2}x faster", 1.0 / speedup);
        }
    }
}

criterion_group!(
    name = heavy_benches;
    config = Criterion::default().sample_size(10).measurement_time(std::time::Duration::from_secs(15));
    targets = heavy_cpu_vs_gpu_battle, multidimensional_heavy_battle
);
criterion_main!(heavy_benches);
