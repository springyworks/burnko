use burn_ndarray::{NdArray, NdArrayDevice};
use burn_wgpu::{Wgpu, WgpuDevice};
use burn_tensor::{Tensor, TensorData, Shape};
use std::time::Instant;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

/// Direct CPU vs GPU comparison benchmark with 1 billion elements
/// Measures real-world performance including all overheads

const GIGA_ELEMENTS: usize = 1_000_000_000; // 1 billion elements
const MEGA_ELEMENTS: usize = 1_000_000;     // 1 million elements
const HUNDRED_MEGA: usize = 100_000_000;    // 100 million elements

type CpuBackend = NdArray<f32>;
type GpuBackend = Wgpu<f32, i32>;

/// Generate analytical test data with predictable cumsum results
fn generate_analytical_data(size: usize) -> Vec<f32> {
    // Pattern: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, ...]
    // Cumsum of first 5: [1, 3, 6, 10, 15]
    // Cumsum of next 5: [16, 18, 21, 25, 30], etc.
    (0..size).map(|i| ((i % 5) + 1) as f32).collect()
}

/// Quick verification of cumsum correctness
fn verify_correctness(input: &[f32], output: &[f32], sample_size: usize, backend_name: &str) -> bool {
    let check_size = sample_size.min(input.len());
    
    for i in 0..check_size.min(100) { // Check first 100 elements
        let expected: f32 = input[0..=i].iter().sum();
        if (output[i] - expected).abs() > 1e-3 {
            println!("❌ {} verification failed at index {}: expected {}, got {}", 
                     backend_name, i, expected, output[i]);
            return false;
        }
    }
    
    println!("✅ {} verification passed", backend_name);
    true
}

/// Head-to-head CPU vs GPU comparison
fn bench_cpu_vs_gpu_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("CPU vs GPU Direct Comparison");
    group.sample_size(10); // Reduce sample size for long tests
    
    let sizes = vec![
        ("1M", MEGA_ELEMENTS),
        ("100M", HUNDRED_MEGA),
        ("1G", GIGA_ELEMENTS),
    ];
    
    let gpu_device = WgpuDevice::default();
    let cpu_device = NdArrayDevice::Cpu;
    
    for (name, size) in sizes {
        // CPU Benchmark
        group.bench_with_input(BenchmarkId::new("CPU", name), &size, |b, &size| {
            let data = generate_analytical_data(size);
            
            b.iter(|| {
                let start = Instant::now();
                
                let tensor: Tensor<CpuBackend, 1> = Tensor::from_data(
                    TensorData::new(data.clone(), Shape::new([size])), &cpu_device
                );
                let result = tensor.cumsum(0);
                let output = result.slice([0..1000]).to_data().to_vec::<f32>().unwrap();
                
                let duration = start.elapsed();
                
                verify_correctness(&data[0..1000], &output, 1000, "CPU");
                
                println!("🖥️  CPU {} elements: {:?}", name, duration);
                
                // Calculate throughput
                let elements_per_sec = size as f64 / duration.as_secs_f64();
                let gb_per_sec = (size * std::mem::size_of::<f32>()) as f64 / duration.as_secs_f64() / 1e9;
                println!("   Throughput: {:.0} elements/sec, {:.2} GB/s", elements_per_sec, gb_per_sec);
                
                duration // Return duration instead of result
            });
        });
        
        // GPU Benchmark
        group.bench_with_input(BenchmarkId::new("GPU", name), &size, |b, &size| {
            let data = generate_analytical_data(size);
            
            b.iter(|| {
                let overall_start = Instant::now();
                
                // Detailed timing breakdown for GPU
                let upload_start = Instant::now();
                let tensor: Tensor<GpuBackend, 1> = Tensor::from_data(
                    TensorData::new(data.clone(), Shape::new([size])), &gpu_device
                );
                let upload_time = upload_start.elapsed();
                
                let compute_start = Instant::now();
                let result = tensor.cumsum(0);
                let compute_time = compute_start.elapsed();
                
                let download_start = Instant::now();
                let output = result.slice([0..1000]).to_data().to_vec::<f32>().unwrap();
                let download_time = download_start.elapsed();
                
                let total_time = overall_start.elapsed();
                
                verify_correctness(&data[0..1000], &output, 1000, "GPU");
                
                // Detailed analysis
                let pcie_time = upload_time + download_time;
                let pcie_percent = (pcie_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0;
                let compute_percent = (compute_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0;
                
                println!("🔥 GPU {} elements: {:?}", name, total_time);
                println!("   📤 Upload: {:?} ({:.1}%)", upload_time, (upload_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0);
                println!("   ⚡ Compute: {:?} ({:.1}%)", compute_time, compute_percent);
                println!("   📥 Download: {:?} ({:.1}%)", download_time, (download_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0);
                println!("   🚌 PCIe Overhead: {:.1}%", pcie_percent);
                
                // Calculate effective throughput
                let elements_per_sec = size as f64 / total_time.as_secs_f64();
                let gb_per_sec = (size * std::mem::size_of::<f32>()) as f64 / total_time.as_secs_f64() / 1e9;
                println!("   Throughput: {:.0} elements/sec, {:.2} GB/s", elements_per_sec, gb_per_sec);
                
                total_time // Return total_time instead of result
            });
        });
    }
    
    group.finish();
}

/// Break-even analysis: find the size where GPU becomes faster
fn bench_breakeven_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("Break-even Analysis");
    group.sample_size(20);
    
    let gpu_device = WgpuDevice::default();
    let cpu_device = NdArrayDevice::Cpu;
    
    // Test various sizes to find GPU advantage point
    let test_sizes = vec![
        1_000,      // 1K
        10_000,     // 10K
        100_000,    // 100K
        1_000_000,  // 1M
        10_000_000, // 10M
        100_000_000, // 100M
    ];
    
    for size in test_sizes {
        let size_name = if size >= 1_000_000 {
            format!("{}M", size / 1_000_000)
        } else if size >= 1_000 {
            format!("{}K", size / 1_000)
        } else {
            format!("{}", size)
        };
        
        group.bench_with_input(BenchmarkId::new("CPU_vs_GPU", &size_name), &size, |b, &size| {
            let data = generate_analytical_data(size);
            
            b.iter(|| {
                println!("⚖️  Break-even test: {} elements", size_name);
                
                // CPU timing
                let cpu_start = Instant::now();
                let cpu_tensor: Tensor<CpuBackend, 1> = Tensor::from_data(
                    TensorData::new(data.clone(), Shape::new([size])), &cpu_device
                );
                let cpu_result = cpu_tensor.cumsum(0);
                let _cpu_output = cpu_result.slice([0..100.min(size)]).to_data().to_vec::<f32>().unwrap();
                let cpu_time = cpu_start.elapsed();
                
                // GPU timing
                let gpu_start = Instant::now();
                let gpu_tensor: Tensor<GpuBackend, 1> = Tensor::from_data(
                    TensorData::new(data.clone(), Shape::new([size])), &gpu_device
                );
                let gpu_result = gpu_tensor.cumsum(0);
                let _gpu_output = gpu_result.slice([0..100.min(size)]).to_data().to_vec::<f32>().unwrap();
                let gpu_time = gpu_start.elapsed();
                
                // Analysis
                let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
                let winner = if gpu_time < cpu_time { "🔥 GPU" } else { "🖥️  CPU" };
                
                println!("   CPU: {:?}", cpu_time);
                println!("   GPU: {:?}", gpu_time);
                println!("   Winner: {} (speedup: {:.2}x)", winner, speedup.abs());
                
                speedup // Return speedup instead of tensors
            });
        });
    }
    
    group.finish();
}

/// Memory-bound vs compute-bound analysis
fn bench_operation_characteristics(c: &mut Criterion) {
    let mut group = c.benchmark_group("Operation Characteristics");
    group.sample_size(10);
    
    let gpu_device = WgpuDevice::default();
    let cpu_device = NdArrayDevice::Cpu;
    let size = HUNDRED_MEGA; // 100M elements for detailed analysis
    
    group.bench_function("cumsum_analysis", |b| {
        let data = generate_analytical_data(size);
        
        b.iter(|| {
            println!("🔬 Operation Characteristics Analysis (100M elements):");
            
            // CPU Analysis
            let cpu_start = Instant::now();
            let cpu_tensor: Tensor<CpuBackend, 1> = Tensor::from_data(
                TensorData::new(data.clone(), Shape::new([size])), &cpu_device
            );
            let cpu_prep_time = cpu_start.elapsed();
            
            let cpu_compute_start = Instant::now();
            let cpu_result = cpu_tensor.cumsum(0);
            let cpu_compute_time = cpu_compute_start.elapsed();
            
            let cpu_extract_start = Instant::now();
            let _cpu_output = cpu_result.slice([0..1000]).to_data().to_vec::<f32>().unwrap();
            let cpu_extract_time = cpu_extract_start.elapsed();
            
            println!("🖥️  CPU Breakdown:");
            println!("   Prep: {:?}", cpu_prep_time);
            println!("   Compute: {:?}", cpu_compute_time);
            println!("   Extract: {:?}", cpu_extract_time);
            
            // GPU Analysis with detailed PCIe breakdown  
            let gpu_upload_start = Instant::now();
            let gpu_tensor: Tensor<GpuBackend, 1> = Tensor::from_data(
                TensorData::new(data.clone(), Shape::new([size])), &gpu_device
            );
            let gpu_upload_time = gpu_upload_start.elapsed();
            
            let gpu_compute_start = Instant::now();
            let gpu_result = gpu_tensor.cumsum(0);
            let gpu_compute_time = gpu_compute_start.elapsed();
            
            let gpu_download_start = Instant::now();
            let _gpu_output = gpu_result.slice([0..1000]).to_data().to_vec::<f32>().unwrap();
            let gpu_download_time = gpu_download_start.elapsed();
            
            println!("🔥 GPU Breakdown:");
            println!("   📤 PCIe Upload: {:?}", gpu_upload_time);
            println!("   ⚡ GPU Compute: {:?}", gpu_compute_time);
            println!("   📥 PCIe Download: {:?}", gpu_download_time);
            
            // Memory bandwidth analysis
            let bytes = size * std::mem::size_of::<f32>();
            let cpu_bandwidth = (bytes as f64) / (cpu_prep_time + cpu_compute_time).as_secs_f64() / 1e9;
            let gpu_compute_bandwidth = (bytes as f64) / gpu_compute_time.as_secs_f64() / 1e9;
            let pcie_upload_bandwidth = (bytes as f64) / gpu_upload_time.as_secs_f64() / 1e9;
            
            println!("📊 Bandwidth Analysis:");
            println!("   CPU: {:.2} GB/s", cpu_bandwidth);
            println!("   GPU Compute: {:.2} GB/s", gpu_compute_bandwidth);
            println!("   PCIe Upload: {:.2} GB/s", pcie_upload_bandwidth);
            
            // Determine bottleneck
            let pcie_total = gpu_upload_time + gpu_download_time;
            if pcie_total > gpu_compute_time {
                println!("🚌 Bottleneck: PCIe transfers ({:.1}% of GPU time)", 
                         (pcie_total.as_secs_f64() / (pcie_total + gpu_compute_time).as_secs_f64()) * 100.0);
            } else {
                println!("⚡ Bottleneck: GPU compute");
            }
            
            gpu_compute_time // Return timing instead of tensors
        });
    });
    
    group.finish();
}

criterion_group!(
    comparison_benches,
    bench_cpu_vs_gpu_direct,
    bench_breakeven_analysis,
    bench_operation_characteristics
);

criterion_main!(comparison_benches);
