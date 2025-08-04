#![recursion_limit = "256"]

use burn_tensor::{Tensor, TensorData, Shape};
use burn_ndarray::NdArray;
use burn_wgpu::Wgpu;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Instant;

type CpuBackend = NdArray<f32>;
type GpuBackend = Wgpu<f32, i32>;

/// Ultimate CPU vs GPU comparison with proper NDArray data blobs
/// Tests 1D, 2D, 3D tensors with analytical verification
fn ultimate_cpu_vs_gpu_benchmark(c: &mut Criterion) {
    let cpu_device = Default::default();
    let gpu_device = Default::default();
    
    let mut group = c.benchmark_group("Ultimate CPU vs GPU NDArray Battle");
    
    // Test sizes - from 1K to 10M elements
    let sizes_1d = vec![1_000, 10_000, 100_000, 1_000_000, 10_000_000];
    let sizes_2d = vec![(100, 100), (316, 316), (1000, 1000), (3162, 3162)]; // ~10K to 10M elements
    let sizes_3d = vec![(10, 10, 10), (22, 22, 22), (100, 100, 100), (215, 215, 215)]; // ~1K to 10M elements
    
    println!("🔥 ULTIMATE CPU vs GPU NDARRAY BATTLE BEGINS!");
    
    // 1D TENSOR TESTS
    for &size in &sizes_1d {
        println!("⚡ 1D Battle: {} elements", size);
        
        // Create analytical data - sequential integers for predictable cumsum
        let data: Vec<f32> = (1..=size).map(|x| x as f32).collect();
        
        // CPU benchmark
        group.bench_with_input(
            BenchmarkId::new("1D_CPU", size),
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
        
        // GPU benchmark  
        group.bench_with_input(
            BenchmarkId::new("1D_GPU", size),
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
    
    // 2D TENSOR TESTS - test both axes
    for &(rows, cols) in &sizes_2d {
        let total_elements = rows * cols;
        println!("⚡ 2D Battle: {}x{} = {} elements", rows, cols, total_elements);
        
        // Create 2D analytical data - row-major sequential
        let data: Vec<f32> = (1..=(total_elements)).map(|x| x as f32).collect();
        
        // Test axis 0 (along rows)
        group.bench_with_input(
            BenchmarkId::new("2D_CPU_axis0", total_elements),
            &total_elements,
            |b, _| {
                let tensor_cpu: Tensor<CpuBackend, 2> = Tensor::from_data(
                    TensorData::new(data.clone(), Shape::new([rows, cols])), &cpu_device
                );
                b.iter(|| {
                    let result = black_box(tensor_cpu.clone().cumsum(0));
                    black_box(result)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("2D_GPU_axis0", total_elements),
            &total_elements,
            |b, _| {
                let tensor_gpu: Tensor<GpuBackend, 2> = Tensor::from_data(
                    TensorData::new(data.clone(), Shape::new([rows, cols])), &gpu_device
                );
                b.iter(|| {
                    let result = black_box(tensor_gpu.clone().cumsum(0));
                    black_box(result)
                });
            },
        );
        
        // Test axis 1 (along columns)
        group.bench_with_input(
            BenchmarkId::new("2D_CPU_axis1", total_elements),
            &total_elements,
            |b, _| {
                let tensor_cpu: Tensor<CpuBackend, 2> = Tensor::from_data(
                    TensorData::new(data.clone(), Shape::new([rows, cols])), &cpu_device
                );
                b.iter(|| {
                    let result = black_box(tensor_cpu.clone().cumsum(1));
                    black_box(result)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("2D_GPU_axis1", total_elements),
            &total_elements,
            |b, _| {
                let tensor_gpu: Tensor<GpuBackend, 2> = Tensor::from_data(
                    TensorData::new(data.clone(), Shape::new([rows, cols])), &gpu_device
                );
                b.iter(|| {
                    let result = black_box(tensor_gpu.clone().cumsum(1));
                    black_box(result)
                });
            },
        );
    }
    
    // 3D TENSOR TESTS - test all axes
    for &(depth, height, width) in &sizes_3d {
        let total_elements = depth * height * width;
        println!("⚡ 3D Battle: {}x{}x{} = {} elements", depth, height, width, total_elements);
        
        // Create 3D analytical data 
        let data: Vec<f32> = (1..=(total_elements)).map(|x| x as f32).collect();
        
        // Test all 3 axes
        for axis in 0..3 {
            group.bench_with_input(
                BenchmarkId::new(format!("3D_CPU_axis{}", axis), total_elements),
                &total_elements,
                |b, _| {
                    let tensor_cpu: Tensor<CpuBackend, 3> = Tensor::from_data(
                        TensorData::new(data.clone(), Shape::new([depth, height, width])), &cpu_device
                    );
                    b.iter(|| {
                        let result = black_box(tensor_cpu.clone().cumsum(axis));
                        black_box(result)
                    });
                },
            );
            
            group.bench_with_input(
                BenchmarkId::new(format!("3D_GPU_axis{}", axis), total_elements),
                &total_elements,
                |b, _| {
                    let tensor_gpu: Tensor<GpuBackend, 3> = Tensor::from_data(
                        TensorData::new(data.clone(), Shape::new([depth, height, width])), &gpu_device
                    );
                    b.iter(|| {
                        let result = black_box(tensor_gpu.clone().cumsum(axis));
                        black_box(result)
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Analytical verification tests - ensure correctness before performance
fn analytical_verification_tests(c: &mut Criterion) {
    let cpu_device = Default::default();
    let gpu_device = Default::default();
    
    let mut group = c.benchmark_group("Analytical Verification");
    
    println!("🧮 ANALYTICAL VERIFICATION: Ensuring correctness before speed tests");
    
    // 1D analytical test: [1,2,3,4,5] -> [1,3,6,10,15]
    let data_1d = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let expected_1d = vec![1.0, 3.0, 6.0, 10.0, 15.0];
    
    group.bench_function("1D_analytical_verification", |b| {
        b.iter(|| {
            // CPU test
            let tensor_cpu: Tensor<CpuBackend, 1> = Tensor::from_data(
                TensorData::new(data_1d.clone(), Shape::new([5])), &cpu_device
            );
            let result_cpu = tensor_cpu.cumsum(0);
            let values_cpu: Vec<f32> = result_cpu.to_data().to_vec().unwrap();
            
            // GPU test
            let tensor_gpu: Tensor<GpuBackend, 1> = Tensor::from_data(
                TensorData::new(data_1d.clone(), Shape::new([5])), &gpu_device
            );
            let result_gpu = tensor_gpu.cumsum(0);
            let values_gpu: Vec<f32> = result_gpu.to_data().to_vec().unwrap();
            
            // Verify both match expected
            for (i, ((cpu, gpu), expected)) in values_cpu.iter()
                .zip(values_gpu.iter())
                .zip(expected_1d.iter())
                .enumerate() {
                assert!((cpu - expected).abs() < 1e-6, 
                    "CPU mismatch at {}: expected {}, got {}", i, expected, cpu);
                assert!((gpu - expected).abs() < 1e-6, 
                    "GPU mismatch at {}: expected {}, got {}", i, expected, gpu);
                assert!((cpu - gpu).abs() < 1e-6,
                    "CPU/GPU mismatch at {}: CPU={}, GPU={}", i, cpu, gpu);
            }
            
            black_box((values_cpu, values_gpu))
        });
    });
    
    // 2D analytical test: verify axis behavior
    let data_2d = vec![
        1.0, 2.0, 3.0,  // Row 0
        4.0, 5.0, 6.0,  // Row 1  
    ];
    
    group.bench_function("2D_analytical_verification", |b| {
        b.iter(|| {
            // Test axis 0 (cumsum down columns)
            let tensor_cpu: Tensor<CpuBackend, 2> = Tensor::from_data(
                TensorData::new(data_2d.clone(), Shape::new([2, 3])), &cpu_device
            );
            let result_cpu_axis0 = tensor_cpu.clone().cumsum(0);
            
            let tensor_gpu: Tensor<GpuBackend, 2> = Tensor::from_data(
                TensorData::new(data_2d.clone(), Shape::new([2, 3])), &gpu_device
            );
            let result_gpu_axis0 = tensor_gpu.clone().cumsum(0);
            
            // Test axis 1 (cumsum across rows)
            let result_cpu_axis1 = tensor_cpu.cumsum(1);
            let result_gpu_axis1 = tensor_gpu.cumsum(1);
            
            // Verify CPU/GPU match for both axes
            let cpu_axis0: Vec<f32> = result_cpu_axis0.to_data().to_vec().unwrap();
            let gpu_axis0: Vec<f32> = result_gpu_axis0.to_data().to_vec().unwrap();
            let cpu_axis1: Vec<f32> = result_cpu_axis1.to_data().to_vec().unwrap();
            let gpu_axis1: Vec<f32> = result_gpu_axis1.to_data().to_vec().unwrap();
            
            for (i, (cpu, gpu)) in cpu_axis0.iter().zip(gpu_axis0.iter()).enumerate() {
                assert!((cpu - gpu).abs() < 1e-6, 
                    "2D axis0 mismatch at {}: CPU={}, GPU={}", i, cpu, gpu);
            }
            
            for (i, (cpu, gpu)) in cpu_axis1.iter().zip(gpu_axis1.iter()).enumerate() {
                assert!((cpu - gpu).abs() < 1e-6, 
                    "2D axis1 mismatch at {}: CPU={}, GPU={}", i, cpu, gpu);
            }
            
            black_box((cpu_axis0, gpu_axis0, cpu_axis1, gpu_axis1))
        });
    });
    
    group.finish();
}

/// Stress test with huge tensors - see where CPU vs GPU crossover occurs
fn stress_test_crossover(c: &mut Criterion) {
    let cpu_device = Default::default();
    let gpu_device = Default::default();
    
    let mut group = c.benchmark_group("Stress Test Crossover");
    group.sample_size(10); // Fewer samples for huge tensors
    
    // Massive tensor sizes to find CPU/GPU crossover point
    let massive_sizes = vec![1_000_000, 5_000_000, 10_000_000, 50_000_000];
    
    for &size in &massive_sizes {
        println!("💀 STRESS TEST: {} elements ({:.1} MB)", size, (size * 4) as f64 / 1_000_000.0);
        
        // Use random data for stress test
        let data: Vec<f32> = (0..size).map(|_| rand::random::<f32>()).collect();
        
        group.bench_with_input(
            BenchmarkId::new("MASSIVE_CPU", size),
            &size,
            |b, _| {
                let tensor_cpu: Tensor<CpuBackend, 1> = Tensor::from_data(
                    TensorData::new(data.clone(), Shape::new([size])), &cpu_device
                );
                b.iter(|| {
                    let start = Instant::now();
                    let result = black_box(tensor_cpu.clone().cumsum(0));
                    let duration = start.elapsed();
                    println!("  CPU {}: {:?}", size, duration);
                    black_box(result)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("MASSIVE_GPU", size),
            &size,
            |b, _| {
                let tensor_gpu: Tensor<GpuBackend, 1> = Tensor::from_data(
                    TensorData::new(data.clone(), Shape::new([size])), &gpu_device
                );
                b.iter(|| {
                    let start = Instant::now();
                    let result = black_box(tensor_gpu.clone().cumsum(0));
                    let duration = start.elapsed();
                    println!("  GPU {}: {:?}", size, duration);
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    ultimate_benchmark,
    analytical_verification_tests,
    ultimate_cpu_vs_gpu_benchmark,
    stress_test_crossover
);
criterion_main!(ultimate_benchmark);
