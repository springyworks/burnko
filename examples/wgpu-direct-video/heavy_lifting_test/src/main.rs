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
    
    // Progressive sizes to find GPU crossover point - pushing to MASSIVE sizes for GPU advantage
    let test_sizes = vec![
        1_000_000,   // 1M - we know CPU wins here (baseline)
        2_000_000,   // 2M - CPU still winning 3.41x
        5_000_000,   // 5M - getting into serious GPU territory
        7_500_000,   // 7.5M - intermediate heavy
        10_000_000,  // 10M - THE BIG TEST! Where GPU should finally shine
        15_000_000,  // 15M - if GPU wins at 10M, push further
        20_000_000,  // 20M - large GPU test
    ];
    
    for &size in &test_sizes {
        println!("\n🚀 Testing {} elements ({:.1} MB)", size, (size * 4) as f64 / 1_000_000.0);
        
        // Create test data
        let data: Vec<f32> = (1..=size).map(|x| (x % 1000) as f32).collect();
        
        // CPU Test with detailed timing
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
        
        // GPU Test with timeout protection
        println!("  Testing GPU...");
        let gpu_start_total = Instant::now();
        
        let start = Instant::now();
        let tensor_gpu: Tensor<GpuBackend, 1> = Tensor::from_data(
            TensorData::new(data.clone(), Shape::new([size])), &gpu_device
        );
        let gpu_creation_time = start.elapsed();
        
        // Check if creation took too long
        if gpu_creation_time.as_secs() > 15 {
            println!("    ⚠️  GPU tensor creation taking too long ({:?}), skipping compute test", gpu_creation_time);
            continue;
        }
        
        let start = Instant::now();
        let result_gpu = tensor_gpu.cumsum(0);
        let gpu_compute_time = start.elapsed();
        
        // Check if compute took too long
        if gpu_compute_time.as_secs() > 30 {
            println!("    ⚠️  GPU compute taking too long ({:?}), skipping retrieval", gpu_compute_time);
            continue;
        }
        
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
        
        // Early exit if GPU is taking way too long
        if gpu_total_time.as_secs() > 45 {
            println!("    ⚠️  GPU taking too long (>45s), stopping progression here");
            break;
        }
        
        // Continue if we found GPU advantage
        if speedup > 1.1 {
            println!("    🎯 Found GPU advantage! Continuing to even larger sizes...");
        }
    }
    
    // Save results to file
    match OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("heavy_benchmark_results.txt") 
    {
        Ok(mut file) => {
            writeln!(file, "Size,CPU_Create_ms,CPU_Compute_ms,CPU_Retrieve_ms,GPU_Create_ms,GPU_Compute_ms,GPU_Retrieve_ms,CPU_Throughput_Melems_sec,GPU_Throughput_Melems_sec,GPU_Speedup").unwrap();
            
            for result in &results {
                file.write_all(result.as_bytes()).unwrap();
            }
            
            println!("\n📊 Results saved to heavy_benchmark_results.txt");
        }
        Err(e) => println!("Failed to save results: {}", e),
    }
    
    println!("🏁 Heavy lifting test complete!");
    println!("\nSUMMARY:");
    for (i, result) in results.iter().enumerate() {
        let parts: Vec<&str> = result.split(',').collect();
        if parts.len() >= 10 {
            let size = parts[0];
            let cpu_throughput = parts[7];
            let gpu_throughput = parts[8];
            let speedup: f64 = parts[9].trim().parse().unwrap_or(0.0);
            
            if speedup > 1.0 {
                println!("  {} elements: GPU wins {:.2}x ({} vs {} Melems/sec)", 
                         size, speedup, gpu_throughput, cpu_throughput);
            } else {
                println!("  {} elements: CPU wins {:.2}x ({} vs {} Melems/sec)", 
                         size, 1.0/speedup, cpu_throughput, gpu_throughput);
            }
        }
    }
}
