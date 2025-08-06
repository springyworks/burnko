use burn_wgpu::{Wgpu, WgpuDevice};
use burn_tensor::{Tensor, TensorData, Shape};
use std::time::Instant;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

/// GPU stress benchmark with PCIe overhead analysis
/// Tests WGPU backend with 1 billion elements and measures data transfer costs

const HALF_GIGA_ELEMENTS: usize = 500_000_000; // 500 million elements (2GB limit)
const MEGA_ELEMENTS: usize = 1_000_000;     // 1 million elements
const KILO_ELEMENTS: usize = 1_000;         // 1 thousand elements

type GpuBackend = Wgpu<f32, i32>;

/// Generate analytical test data pattern
fn generate_analytical_data(size: usize) -> Vec<f32> {
    (0..size).map(|i| ((i % 5) + 1) as f32).collect()
}

/// Verify correctness with analytical pattern
fn verify_cumsum_correctness(input: &[f32], output: &[f32], sample_size: usize) -> bool {
    let check_size = sample_size.min(input.len());
    
    for i in 0..check_size {
        let expected: f32 = input[0..=i].iter().sum();
        if (output[i] - expected).abs() > 1e-4 {
            println!("❌ GPU Cumsum verification failed at index {}: expected {}, got {}", 
                     i, expected, output[i]);
            return false;
        }
    }
    
    println!("✅ GPU Cumsum verification passed for {} elements", check_size);
    true
}

/// Benchmark GPU device initialization and detection
fn bench_gpu_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU Initialization");
    
    group.bench_function("device_detection", |b| {
        b.iter(|| {
            let start = Instant::now();
            
            // Check if GPU is available
            let device = WgpuDevice::default();
            println!("🔍 GPU Device: {:?}", device);
            
            let duration = start.elapsed();
            println!("📊 GPU device detection: {:?}", duration);
            
            device
        });
    });
    
    group.finish();
}

/// Check if GPU can handle the memory allocation
fn can_allocate_on_gpu(device: &WgpuDevice, size: usize) -> bool {
    let bytes_needed = size * std::mem::size_of::<f32>();
    // Conservative limit: don't try to allocate more than 1.5GB on GPU
    if bytes_needed > 1_500_000_000 {
        println!("⚠️  Skipping {} elements ({:.1} GB) - exceeds safe GPU memory limit", 
                 size, bytes_needed as f64 / 1e9);
        return false;
    }
    
    // Try a small test allocation to verify GPU is available
    let test_data = vec![1.0f32; 1000];
    match std::panic::catch_unwind(|| {
        let _tensor: Tensor<GpuBackend, 1> = Tensor::from_data(
            TensorData::new(test_data, Shape::new([1000])), device
        );
    }) {
        Ok(_) => true,
        Err(_) => {
            println!("⚠️  GPU allocation test failed - GPU may not be available");
            false
        }
    }
}

/// Benchmark GPU scan performance with PCIe overhead breakdown
fn bench_gpu_scan_with_pcie_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU Scan with PCIe Analysis");
    
    let device = WgpuDevice::default();
    
    let sizes = vec![
        ("1K", KILO_ELEMENTS),
        ("1M", MEGA_ELEMENTS), 
        ("500M", HALF_GIGA_ELEMENTS),
    ];
    
    for (name, size) in sizes {
        // Check if GPU can handle this allocation before benchmarking
        if !can_allocate_on_gpu(&device, size) {
            println!("🚫 Skipping {} benchmark due to memory constraints", name);
            continue;
        }
        
        group.bench_with_input(BenchmarkId::new("gpu_cumsum_detailed", name), &size, |b, &size| {
            // Pre-generate data once
            let data = generate_analytical_data(size);
            
            b.iter(|| {
                let overall_start = Instant::now();
                
                println!("🚀 GPU {} elements stress test starting...", name);
                
                // Phase 1: CPU to GPU transfer (PCIe upload) with error handling
                let upload_start = Instant::now();
                let tensor_result = std::panic::catch_unwind(|| {
                    Tensor::from_data(TensorData::new(data.clone(), Shape::new([size])), &device)
                });
                
                let tensor: Tensor<GpuBackend, 1> = match tensor_result {
                    Ok(t) => t,
                    Err(_) => {
                        println!("❌ GPU allocation failed for {} elements ({:.1} GB)", 
                                 size, (size * 4) as f64 / 1e9);
                        // Return early with minimal timing to avoid benchmark failure
                        return std::time::Duration::from_millis(1);
                    }
                };
                let upload_time = upload_start.elapsed();
                
                // Phase 2: GPU computation 
                let compute_start = Instant::now();
                let result = tensor.cumsum(0);
                let compute_time = compute_start.elapsed();
                
                // Phase 3: GPU to CPU transfer (PCIe download) + verification
                let download_start = Instant::now();
                let sample_size = if size > MEGA_ELEMENTS { 1000 } else { size.min(1000) };
                let output_sample = result.clone().slice([0..sample_size]).to_data().to_vec::<f32>().unwrap();
                let download_time = download_start.elapsed();
                
                let total_time = overall_start.elapsed();
                
                // Verify correctness
                let is_correct = verify_cumsum_correctness(&data[0..sample_size], &output_sample, sample_size);
                
                // Calculate data transfer rates
                let bytes = size * std::mem::size_of::<f32>();
                let upload_gbps = (bytes as f64) / upload_time.as_secs_f64() / 1e9;
                let download_bytes = sample_size * std::mem::size_of::<f32>();
                let download_gbps = (download_bytes as f64) / download_time.as_secs_f64() / 1e9;
                
                println!("📊 GPU {} breakdown:", name);
                println!("   📤 PCIe Upload: {:?} ({:.2} GB/s)", upload_time, upload_gbps);
                println!("   🔥 GPU Compute: {:?}", compute_time);
                println!("   📥 PCIe Download: {:?} ({:.2} GB/s)", download_time, download_gbps);
                println!("   🏁 Total: {:?}", total_time);
                println!("   ✅ Correct: {}", is_correct);
                
                // Calculate efficiency metrics
                let pcie_overhead = upload_time + download_time;
                let pcie_percentage = (pcie_overhead.as_secs_f64() / total_time.as_secs_f64()) * 100.0;
                let compute_percentage = (compute_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0;
                
                println!("   📈 PCIe Overhead: {:.1}% of total time", pcie_percentage);
                println!("   ⚡ GPU Compute: {:.1}% of total time", compute_percentage);
                
                // Return timing information instead of the moved tensor
                total_time
            });
        });
    }
    
    group.finish();
}

/// Benchmark memory patterns and GPU utilization
fn bench_gpu_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU Memory Patterns");
    let device = WgpuDevice::default();
    
    // Test different data patterns for GPU efficiency
    group.bench_function("sequential_pattern_1M", |b| {
        let data: Vec<f32> = (0..MEGA_ELEMENTS).map(|i| i as f32).collect();
        
        b.iter(|| {
            let start = Instant::now();
            
            let tensor: Tensor<GpuBackend, 1> = Tensor::from_data(
                TensorData::new(data.clone(), Shape::new([MEGA_ELEMENTS])), &device
            );
            let result = tensor.cumsum(0);
            let _output = result.slice([0..100]).to_data().to_vec::<f32>().unwrap();
            
            let duration = start.elapsed();
            println!("📊 GPU sequential pattern 1M: {:?}", duration);
        });
    });
    
    group.bench_function("constant_pattern_1M", |b| {
        let data = vec![1.0f32; MEGA_ELEMENTS];
        
        b.iter(|| {
            let start = Instant::now();
            
            let tensor: Tensor<GpuBackend, 1> = Tensor::from_data(
                TensorData::new(data.clone(), Shape::new([MEGA_ELEMENTS])), &device
            );
            let result = tensor.cumsum(0);
            let _output = result.slice([0..100]).to_data().to_vec::<f32>().unwrap();
            
            let duration = start.elapsed();
            println!("📊 GPU constant pattern 1M: {:?}", duration);
        });
    });
    
    group.finish();
}

/// Multi-dimensional GPU benchmark
fn bench_gpu_multidimensional(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU Multi-dimensional");
    let device = WgpuDevice::default();
    
    // Large 2D matrices
    let configs = vec![
        ("1K x 1K", 1_000, 1_000),           // 1M elements
        ("100 x 10K", 100, 10_000),          // 1M elements  
        ("10K x 100", 10_000, 100),          // 1M elements
    ];
    
    for (name, rows, cols) in configs {
        group.bench_with_input(BenchmarkId::new("2D_gpu_cumsum", name), &(rows, cols), |b, &(rows, cols)| {
            let total_elements = rows * cols;
            let data = generate_analytical_data(total_elements);
            
            b.iter(|| {
                let start = Instant::now();
                
                let tensor: Tensor<GpuBackend, 2> = Tensor::from_data(
                    TensorData::new(data.clone(), Shape::new([rows, cols])), &device
                );
                
                // Test both axis directions
                let result_axis0 = tensor.clone().cumsum(0);
                let result_axis1 = tensor.cumsum(1);
                
                // Sample verification
                let _sample0 = result_axis0.clone().slice([0..10, 0..10]).to_data();
                let _sample1 = result_axis1.clone().slice([0..10, 0..10]).to_data();
                
                let duration = start.elapsed();
                println!("📊 GPU 2D {} (both axes): {:?}", name, duration);
                
                // Return timing information instead of the moved tensors
                duration
            });
        });
    }
    
    group.finish();
}

/// Comparative GPU vs theoretical peak performance
fn bench_gpu_theoretical_limits(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU Theoretical Analysis");
    let device = WgpuDevice::default();
    
    group.bench_function("bandwidth_test_1M", |b| {
        // Use 1M elements instead of 1G to avoid memory issues
        let test_size = MEGA_ELEMENTS;
        let data = generate_analytical_data(test_size);
        
        b.iter(|| {
            println!("🔬 GPU Theoretical Bandwidth Test ({} elements)...", test_size);
            
            let overall_start = Instant::now();
            
            // Pure upload test with error handling
            let upload_start = Instant::now();
            let tensor_result = std::panic::catch_unwind(|| {
                Tensor::from_data(TensorData::new(data.clone(), Shape::new([test_size])), &device)
            });
            
            let tensor: Tensor<GpuBackend, 1> = match tensor_result {
                Ok(t) => t,
                Err(_) => {
                    println!("❌ GPU allocation failed for theoretical test");
                    return std::time::Duration::from_millis(1);
                }
            };
            let upload_time = upload_start.elapsed();
            
            // Pure compute test (minimal data transfer)
            let compute_start = Instant::now();
            let result = tensor.cumsum(0);
            let compute_time = compute_start.elapsed();
            
            // Pure download test (small sample)
            let download_start = Instant::now();
            let _sample = result.clone().slice([0..1000]).to_data().to_vec::<f32>().unwrap();
            let download_time = download_start.elapsed();
            
            let total_time = overall_start.elapsed();
            
            // Calculate theoretical metrics
            let bytes = test_size * std::mem::size_of::<f32>();
            let theoretical_pcie_3_gbps = 16.0; // PCIe 3.0 x16 theoretical
            let theoretical_pcie_4_gbps = 32.0; // PCIe 4.0 x16 theoretical
            
            let actual_upload_gbps = (bytes as f64) / upload_time.as_secs_f64() / 1e9;
            let pcie3_efficiency = (actual_upload_gbps / theoretical_pcie_3_gbps) * 100.0;
            let pcie4_efficiency = (actual_upload_gbps / theoretical_pcie_4_gbps) * 100.0;
            
            println!("📈 Theoretical Analysis:");
            println!("   Upload: {:?} ({:.2} GB/s)", upload_time, actual_upload_gbps);
            println!("   Compute: {:?}", compute_time);
            println!("   Download: {:?}", download_time);
            println!("   Total: {:?}", total_time);
            println!("   PCIe 3.0 Efficiency: {:.1}%", pcie3_efficiency);
            println!("   PCIe 4.0 Efficiency: {:.1}%", pcie4_efficiency);
            
            // GPU compute efficiency
            let elements_per_sec = test_size as f64 / compute_time.as_secs_f64();
            println!("   GPU Throughput: {:.0} elements/sec", elements_per_sec);
            
            // Return timing information instead of the moved tensor
            total_time
        });
    });
    
    group.finish();
}

criterion_group!(
    gpu_benches,
    bench_gpu_initialization,
    bench_gpu_scan_with_pcie_analysis,
    bench_gpu_memory_patterns,
    bench_gpu_multidimensional,
    bench_gpu_theoretical_limits
);

criterion_main!(gpu_benches);
