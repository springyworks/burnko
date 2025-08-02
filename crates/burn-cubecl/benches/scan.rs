use burn_wgpu::Wgpu;
use burn_tensor::{Tensor, Distribution};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

type WgpuBackend = Wgpu<f32>;

/// Test sizes that demonstrate the scan performance characteristics
const SCAN_SIZES: &[usize] = &[
    16,    // Small - efficient parallel scan
    64,    // Medium - transition point  
    256,   // Large - cube size boundary
    1024,  // Very large - chunked processing
    4096,  // Ultra large - performance validation
];

/// Benchmark GPU scan implementation
/// Tests the improved scan algorithm with no automatic switching
fn benchmark_gpu_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_scan");
    group.measurement_time(Duration::from_secs(10));
    
    for &size in SCAN_SIZES {
        group.throughput(Throughput::Elements(size as u64));
        
        let device = Default::default();
        let tensor: Tensor<WgpuBackend, 1> = Tensor::random(
            [size],
            Distribution::Uniform(1.0, 2.0),
            &device,
        );
        
        group.bench_with_input(
            BenchmarkId::new("cumsum", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let config = burn_tensor::ops::ScanConfig::new(
                        burn_tensor::ops::ScanOp::Add, 
                        0
                    );
                    let result = black_box(tensor.clone().scan(config));
                    // Force GPU synchronization
                    let _ = result.to_data();
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark scan operations for different scan operations (Add, Mul, Max, Min)
fn benchmark_scan_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan_operations");
    group.measurement_time(Duration::from_secs(8));
    
    let size = 1024; // Fixed size for operation comparison
    let device = Default::default();
    
    let operations = [
        ("Add", burn_tensor::ops::ScanOp::Add),
        ("Mul", burn_tensor::ops::ScanOp::Mul), 
        ("Max", burn_tensor::ops::ScanOp::Max),
        ("Min", burn_tensor::ops::ScanOp::Min),
    ];
    
    for (op_name, scan_op) in operations {
        let tensor: Tensor<WgpuBackend, 1> = Tensor::random(
            [size],
            Distribution::Uniform(1.0, 2.0),
            &device,
        );
        
        group.bench_with_input(
            BenchmarkId::new(op_name, size),
            &size,
            |b, _| {
                b.iter(|| {
                    let config = burn_tensor::ops::ScanConfig::new(scan_op, 0);
                    let result = black_box(tensor.clone().scan(config));
                    let _ = result.to_data();
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark multi-dimensional scan along different axes
fn benchmark_multidimensional_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("multidimensional_scan");
    group.measurement_time(Duration::from_secs(8));
    
    let device = Default::default();
    
    // Test different tensor shapes and scan dimensions
    let test_configs = [
        ("2D_axis0", [64, 64], 0),
        ("2D_axis1", [64, 64], 1),
    ];
    
    for (name, shape, axis) in test_configs {
        let tensor: Tensor<WgpuBackend, 2> = Tensor::random(
            shape,
            Distribution::Uniform(1.0, 2.0),
            &device,
        );
        
        group.bench_with_input(
            BenchmarkId::new(name, shape.iter().product::<usize>()),
            &axis,
            |b, &scan_axis| {
                b.iter(|| {
                    let config = burn_tensor::ops::ScanConfig::new(
                        burn_tensor::ops::ScanOp::Add, 
                        scan_axis
                    );
                    let result = black_box(tensor.clone().scan(config));
                    let _ = result.to_data();
                })
            },
        );
    }
    
    // Test 3D tensors
    let test_configs_3d = [
        ("3D_axis0", [16, 16, 16], 0),
        ("3D_axis1", [16, 16, 16], 1), 
        ("3D_axis2", [16, 16, 16], 2),
    ];
    
    for (name, shape, axis) in test_configs_3d {
        let tensor: Tensor<WgpuBackend, 3> = Tensor::random(
            shape,
            Distribution::Uniform(1.0, 2.0),
            &device,
        );
        
        group.bench_with_input(
            BenchmarkId::new(name, shape.iter().product::<usize>()),
            &axis,
            |b, &scan_axis| {
                b.iter(|| {
                    let config = burn_tensor::ops::ScanConfig::new(
                        burn_tensor::ops::ScanOp::Add, 
                        scan_axis
                    );
                    let result = black_box(tensor.clone().scan(config));
                    let _ = result.to_data();
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(30);
    targets = 
        benchmark_gpu_scan,
        benchmark_scan_operations,
        benchmark_multidimensional_scan,
);
criterion_main!(benches);
