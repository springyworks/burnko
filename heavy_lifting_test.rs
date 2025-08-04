#![recursion_limit = "256"]

use burn_tensor::{Tensor, TensorData, Shape};
use burn_ndarray::NdArray;
use burn_wgpu::Wgpu;
use std::time::Instant;
use std::fs::OpenOptions;
use std::io::Write;

type CpuBackend = NdArray<f32>;
type GpuBackend = Wgpu<f32, i32>;

fn main() {
    println!("🔥 TARGETED CPU vs GPU HEAVY LIFTING TEST");
    println!("This will save results to heavy_benchmark_results.txt");
    
    let cpu_device = Default::default();
    let gpu_device = Default::default();
    
    let mut results = Vec::new();
    
    // Progressive sizes to find GPU crossover point
    let test_sizes = vec![
        100_000,     // 100K - we know CPU wins here
        500_000,     // 500K - getting larger
        1_000_000,   // 1M - where things might change
        2_000_000,   // 2M - definitely should show GPU advantage if it exists
        5_000_000,   // 5M - heavy lifting territory
    ];
    
    for &size in &test_sizes {
        println!("\n🚀 Testing {} elements ({:.1} MB)", size, (size * 4) as f64 / 1_000_000.0);
        
        // Create test data
        let data: Vec<f32> = (1..=size).map(|x| (x % 1000) as f32).collect();
        
        // CPU Test with timeout protection
        println!("  Testing CPU...");
        let start = Instant::now();
        let tensor_cpu: Tensor<CpuBackend, 1> = Tensor::from_data(
            TensorData::new(data.clone(), Shape::new([size])), &cpu_device
        );
        let cpu_creation_time = start.elapsed();
        
        let start = Instant::now();
        let result_cpu = tensor_cpu.cumsum(0);
        let cpu_compute_time = start.elapsed();
        
        let start = Instant::now();
        let _cpu_data: Vec<f32> = result_cpu.to_data().to_vec().unwrap();
        let cpu_retrieval_time = start.elapsed();
        
        let cpu_total_time = cpu_creation_time + cpu_compute_time + cpu_retrieval_time;
        let cpu_throughput = size as f64 / cpu_total_time.as_secs_f64() / 1_000_000.0;
        
        println!("    CPU: Create {:?} + Compute {:?} + Retrieve {:?} = Total {:?}", 
                 cpu_creation_time, cpu_compute_time, cpu_retrieval_time, cpu_total_time);
        println!("    CPU: {:.2} Melems/sec", cpu_throughput);
        
        // GPU Test with timeout protection and detailed timing
        println!("  Testing GPU...");
        let start = Instant::now();
        let tensor_gpu: Tensor<GpuBackend, 1> = Tensor::from_data(
            TensorData::new(data.clone(), Shape::new([size])), &gpu_device
        );
        let gpu_creation_time = start.elapsed();
        
        let start = Instant::now();
        let result_gpu = tensor_gpu.cumsum(0);
        let gpu_compute_time = start.elapsed();
        
        let start = Instant::now();
        let _gpu_data: Vec<f32> = result_gpu.to_data().to_vec().unwrap();
        let gpu_retrieval_time = start.elapsed();
        
        let gpu_total_time = gpu_creation_time + gpu_compute_time + gpu_retrieval_time;
        let gpu_throughput = size as f64 / gpu_total_time.as_secs_f64() / 1_000_000.0;
        
        println!("    GPU: Create {:?} + Compute {:?} + Retrieve {:?} = Total {:?}", 
                 gpu_creation_time, gpu_compute_time, gpu_retrieval_time, gpu_total_time);
        println!("    GPU: {:.2} Melems/sec", gpu_throughput);
        
        // Calculate speedup
        let speedup = cpu_total_time.as_secs_f64() / gpu_total_time.as_secs_f64();
        if speedup > 1.0 {
            println!("    🏆 GPU is {:.2}x faster", speedup);
        } else {
            println!("    🏆 CPU is {:.2}x faster", 1.0 / speedup);
        }
        
        // Store results
        let result_line = format!(
            "{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2},{:.2},{:.2}\n",
            size,
            cpu_creation_time.as_secs_f64() * 1000.0,
            cpu_compute_time.as_secs_f64() * 1000.0,
            cpu_retrieval_time.as_secs_f64() * 1000.0,
            gpu_creation_time.as_secs_f64() * 1000.0,
            gpu_compute_time.as_secs_f64() * 1000.0,
            gpu_retrieval_time.as_secs_f64() * 1000.0,
            cpu_throughput,
            gpu_throughput,
            speedup
        );
        results.push(result_line);
        
        // Early exit if GPU is taking too long (> 10 seconds total)
        if gpu_total_time.as_secs() > 10 {
            println!("    ⚠️  GPU taking too long (>10s), stopping here");
            break;
        }
        
        // Early exit if we found GPU advantage
        if speedup > 1.2 {
            println!("    🎯 Found GPU advantage! Continuing to larger sizes...");
        }
    }
    
    // Save results to file
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("heavy_benchmark_results.txt")
        .expect("Failed to create results file");
    
    writeln!(file, "Size,CPU_Create_ms,CPU_Compute_ms,CPU_Retrieve_ms,GPU_Create_ms,GPU_Compute_ms,GPU_Retrieve_ms,CPU_Throughput_Melems_sec,GPU_Throughput_Melems_sec,GPU_Speedup").unwrap();
    
    for result in &results {
        file.write_all(result.as_bytes()).unwrap();
    }
    
    println!("\n📊 Results saved to heavy_benchmark_results.txt");
    println!("🏁 Heavy lifting test complete!");
}
