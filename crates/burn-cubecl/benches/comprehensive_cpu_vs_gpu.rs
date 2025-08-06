#![recursion_limit = "256"]

use burn_ndarray::NdArray;
use burn_wgpu::Wgpu;
use burn_tensor::{Tensor, Distribution, activation::relu, activation::softmax};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

type CpuBackend = NdArray<f32>;
type GpuBackend = Wgpu<f32>;

/// Standard tensor sizes for benchmarking - small to large
#[allow(dead_code)]
const SMALL_SIZES: &[usize] = &[64, 128, 256];
const MEDIUM_SIZES: &[usize] = &[512, 1024, 2048];  
#[allow(dead_code)]
const LARGE_SIZES: &[usize] = &[4096, 8192, 16384];

/// Matrix sizes for intensive operations (batch_size, input_dim, output_dim)
const MATRIX_SIZES: &[(usize, usize, usize)] = &[
    (32, 256, 256),     // Small batch, medium matrices
    (64, 512, 512),     // Medium batch, medium matrices
    (128, 1024, 1024),  // Large batch, large matrices
    (256, 2048, 1024),  // Very large - where GPU should shine
];

/// Element-wise operations benchmark - CPU vs GPU
fn bench_elementwise_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_operations");
    group.measurement_time(Duration::from_secs(10));
    
    for &size in MEDIUM_SIZES {
        let total_elements = size * size;
        group.throughput(Throughput::Elements(total_elements as u64));
        
        // CPU setup
        let cpu_device = Default::default();
        let cpu_tensor1: Tensor<CpuBackend, 2> = Tensor::random([size, size], Distribution::Uniform(-1.0, 1.0), &cpu_device);
        let cpu_tensor2: Tensor<CpuBackend, 2> = Tensor::random([size, size], Distribution::Uniform(-1.0, 1.0), &cpu_device);
        
        // GPU setup  
        let gpu_device = Default::default();
        let gpu_tensor1: Tensor<GpuBackend, 2> = Tensor::random([size, size], Distribution::Uniform(-1.0, 1.0), &gpu_device);
        let gpu_tensor2: Tensor<GpuBackend, 2> = Tensor::random([size, size], Distribution::Uniform(-1.0, 1.0), &gpu_device);
        
        // Addition benchmarks
        group.bench_with_input(BenchmarkId::new("add_cpu", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(cpu_tensor1.clone() + cpu_tensor2.clone());
                let _ = result.to_data(); // Force computation
            })
        });
        
        group.bench_with_input(BenchmarkId::new("add_gpu", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(gpu_tensor1.clone() + gpu_tensor2.clone());
                let _ = result.to_data(); // Force computation
            })
        });
        
        // Multiplication benchmarks
        group.bench_with_input(BenchmarkId::new("mul_cpu", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(cpu_tensor1.clone() * cpu_tensor2.clone());
                let _ = result.to_data();
            })
        });
        
        group.bench_with_input(BenchmarkId::new("mul_gpu", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(gpu_tensor1.clone() * gpu_tensor2.clone());
                let _ = result.to_data();
            })
        });
    }
    
    group.finish();
}

/// Matrix multiplication benchmark - GPU vs CPU comparison
fn bench_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_operations");
    group.measurement_time(Duration::from_secs(15));
    
    for &(batch_size, input_dim, output_dim) in MATRIX_SIZES {
        let ops_count = batch_size * input_dim * output_dim * 2; // Approximate FLOPS
        group.throughput(Throughput::Elements(ops_count as u64));
        
        // CPU setup
        let cpu_device = Default::default();
        let cpu_input: Tensor<CpuBackend, 2> = Tensor::random([batch_size, input_dim], Distribution::Uniform(-1.0, 1.0), &cpu_device);
        let cpu_weight: Tensor<CpuBackend, 2> = Tensor::random([input_dim, output_dim], Distribution::Uniform(-1.0, 1.0), &cpu_device);
        
        // GPU setup
        let gpu_device = Default::default();
        let gpu_input: Tensor<GpuBackend, 2> = Tensor::random([batch_size, input_dim], Distribution::Uniform(-1.0, 1.0), &gpu_device);
        let gpu_weight: Tensor<GpuBackend, 2> = Tensor::random([input_dim, output_dim], Distribution::Uniform(-1.0, 1.0), &gpu_device);
        
        let size_label = format!("{}x{}x{}", batch_size, input_dim, output_dim);
        
        // Matrix multiplication benchmarks
        group.bench_with_input(BenchmarkId::new("matmul_cpu", &size_label), &size_label, |b, _| {
            b.iter(|| {
                let result = black_box(cpu_input.clone().matmul(cpu_weight.clone()));
                let _ = result.to_data();
            })
        });
        
        group.bench_with_input(BenchmarkId::new("matmul_gpu", &size_label), &size_label, |b, _| {
            b.iter(|| {
                let result = black_box(gpu_input.clone().matmul(gpu_weight.clone()));
                let _ = result.to_data();
            })
        });
    }
    
    group.finish();
}

/// Activation functions benchmark - ReLU and Softmax
fn bench_activation_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_operations");
    group.measurement_time(Duration::from_secs(8));
    
    for &size in MEDIUM_SIZES {
        let total_elements = size * size;
        group.throughput(Throughput::Elements(total_elements as u64));
        
        // CPU setup
        let cpu_device = Default::default();
        let cpu_tensor: Tensor<CpuBackend, 2> = Tensor::random([size, size], Distribution::Uniform(-2.0, 2.0), &cpu_device);
        
        // GPU setup
        let gpu_device = Default::default();
        let gpu_tensor: Tensor<GpuBackend, 2> = Tensor::random([size, size], Distribution::Uniform(-2.0, 2.0), &gpu_device);
        
        // ReLU benchmarks
        group.bench_with_input(BenchmarkId::new("relu_cpu", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(relu(cpu_tensor.clone()));
                let _ = result.to_data();
            })
        });
        
        group.bench_with_input(BenchmarkId::new("relu_gpu", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(relu(gpu_tensor.clone()));
                let _ = result.to_data();
            })
        });
        
        // Softmax benchmarks (along last dimension)
        group.bench_with_input(BenchmarkId::new("softmax_cpu", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(softmax(cpu_tensor.clone(), 1));
                let _ = result.to_data();
            })
        });
        
        group.bench_with_input(BenchmarkId::new("softmax_gpu", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(softmax(gpu_tensor.clone(), 1));
                let _ = result.to_data();
            })
        });
    }
    
    group.finish();
}

/// Reduction operations benchmark - Sum, Mean across dimensions
fn bench_reduction_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_operations");
    group.measurement_time(Duration::from_secs(8));
    
    for &size in MEDIUM_SIZES {
        let total_elements = size * size;
        group.throughput(Throughput::Elements(total_elements as u64));
        
        // CPU setup
        let cpu_device = Default::default();
        let cpu_tensor: Tensor<CpuBackend, 2> = Tensor::random([size, size], Distribution::Uniform(0.0, 1.0), &cpu_device);
        
        // GPU setup
        let gpu_device = Default::default();
        let gpu_tensor: Tensor<GpuBackend, 2> = Tensor::random([size, size], Distribution::Uniform(0.0, 1.0), &gpu_device);
        
        // Sum all elements
        group.bench_with_input(BenchmarkId::new("sum_all_cpu", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(cpu_tensor.clone().sum());
                let _ = result.to_data();
            })
        });
        
        group.bench_with_input(BenchmarkId::new("sum_all_gpu", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(gpu_tensor.clone().sum());
                let _ = result.to_data();
            })
        });
        
        // Sum along dimension 0
        group.bench_with_input(BenchmarkId::new("sum_dim0_cpu", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(cpu_tensor.clone().sum_dim(0));
                let _ = result.to_data();
            })
        });
        
        group.bench_with_input(BenchmarkId::new("sum_dim0_gpu", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(gpu_tensor.clone().sum_dim(0));
                let _ = result.to_data();
            })
        });
        
        // Mean along dimension 1
        group.bench_with_input(BenchmarkId::new("mean_dim1_cpu", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(cpu_tensor.clone().mean_dim(1));
                let _ = result.to_data();
            })
        });
        
        group.bench_with_input(BenchmarkId::new("mean_dim1_gpu", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(gpu_tensor.clone().mean_dim(1));
                let _ = result.to_data();
            })
        });
    }
    
    group.finish();
}

/// Intensive compute workload - Complex operations chain
fn bench_intensive_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("intensive_compute");
    group.measurement_time(Duration::from_secs(15));
    
    for &size in &[512, 1024, 2048] {
        let total_ops = size * size * 10; // Approximate operation count
        group.throughput(Throughput::Elements(total_ops as u64));
        
        // CPU setup
        let cpu_device = Default::default();
        let cpu_input: Tensor<CpuBackend, 2> = Tensor::random([size, size], Distribution::Uniform(-1.0, 1.0), &cpu_device);
        let cpu_weight: Tensor<CpuBackend, 2> = Tensor::random([size, size], Distribution::Uniform(-1.0, 1.0), &cpu_device);
        
        // GPU setup
        let gpu_device = Default::default();
        let gpu_input: Tensor<GpuBackend, 2> = Tensor::random([size, size], Distribution::Uniform(-1.0, 1.0), &gpu_device);
        let gpu_weight: Tensor<GpuBackend, 2> = Tensor::random([size, size], Distribution::Uniform(-1.0, 1.0), &gpu_device);
        
        // Complex computation: matmul + relu + softmax + sum
        group.bench_with_input(BenchmarkId::new("intensive_cpu", size), &size, |b, _| {
            b.iter(|| {
                let result = cpu_input.clone()
                    .matmul(cpu_weight.clone())
                    .add_scalar(0.1);
                let result = relu(result);
                let result = softmax(result, 1);
                let result = black_box(result.sum_dim(0));
                let _ = result.to_data();
            })
        });
        
        group.bench_with_input(BenchmarkId::new("intensive_gpu", size), &size, |b, _| {
            b.iter(|| {
                let result = gpu_input.clone()
                    .matmul(gpu_weight.clone())
                    .add_scalar(0.1);
                let result = relu(result);
                let result = softmax(result, 1);
                let result = black_box(result.sum_dim(0));
                let _ = result.to_data();
            })
        });
    }
    
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(20)
        .warm_up_time(Duration::from_secs(3));
    targets = 
        bench_elementwise_ops,
        bench_matrix_operations,
        bench_activation_ops,
        bench_reduction_ops,
        bench_intensive_compute,
);

criterion_main!(benches);
