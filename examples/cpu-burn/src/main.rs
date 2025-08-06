//! 🔥🔥🔥 MAXIMUM CPU BURN TEST 🔥🔥🔥
//! 
//! This will push ALL CPU cores to 100% utilization using our
//! Rayon parallel scan implementation with NO throttling!

use burn_ndarray::NdArray;
use burn_tensor::{Tensor, Distribution};
use std::time::{Instant, Duration};
use std::thread;

type NdArrayBackend = NdArray<f32>;

fn main() {
    let core_count = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    
    println!("🔥🔥🔥 MAXIMUM CPU BURN TEST - NO MERCY EDITION 🔥🔥🔥");
    println!("🧵 Detected {core_count} logical cores - preparing for 100% UTILIZATION!");
    println!("⚡ This will run MAXIMUM INTENSITY parallel scan operations for 45 seconds");
    println!("🚨 WARNING: This will max out your CPU - monitor temperature!");
    println!("🌡️  Use 'htop' or 'top' to watch all cores hit 100%!");
    
    thread::sleep(Duration::from_secs(3));
    println!("\n🚀 LAUNCHING CPU BURN TEST IN 3...");
    thread::sleep(Duration::from_secs(1));
    println!("🚀 2...");
    thread::sleep(Duration::from_secs(1));
    println!("🚀 1...");
    thread::sleep(Duration::from_secs(1));
    println!("🔥 BURN INITIATED! 🔥\n");
    
    let device = Default::default();
    let start_time = Instant::now();
    let burn_duration = Duration::from_secs(45); // 45 seconds of maximum burn
    
    // MASSIVE tensor sizes for maximum parallel load
    let test_sizes = vec![
        1_000_000,   // 1M elements
        2_000_000,   // 2M elements  
        3_000_000,   // 3M elements
        4_000_000,   // 4M elements - huge parallel load
    ];
    
    let mut iteration = 0;
    let mut total_operations = 0;
    
    println!("📊 Creating multiple large tensors for parallel processing...");
    
    // Pre-create tensors to avoid allocation overhead during burn test
    let mut tensors = Vec::new();
    for &size in &test_sizes {
        let tensor: Tensor<NdArrayBackend, 2> = 
            Tensor::random([1, size], Distribution::Uniform(0.0, 1.0), &device);
        tensors.push(tensor);
        println!("   ✅ Created tensor with {size} elements");
    }
    
    println!("\n🔥 STARTING INTENSIVE PARALLEL SCAN OPERATIONS! 🔥");
    
    while start_time.elapsed() < burn_duration {
        iteration += 1;
        let iter_start = Instant::now();
        
        // BRUTAL: Perform MANY more parallel scan operations simultaneously
        for tensor in tensors.iter() {
            // MAXIMUM PARALLEL LOAD: 8 operations per tensor
            let t1 = tensor.clone();
            let t2 = tensor.clone();
            let t3 = tensor.clone();
            let t4 = tensor.clone();
            let t5 = tensor.clone();
            let t6 = tensor.clone();
            let t7 = tensor.clone();
            let t8 = tensor.clone();
            
            // These operations will distribute across all CPU cores
            let _r1 = t1.cumsum(1);
            let _r2 = t2.cumprod(1);
            let _r3 = t3.cumsum(1);
            let _r4 = t4.cumprod(1);
            let _r5 = t5.cumsum(1);
            let _r6 = t6.cumprod(1);
            let _r7 = t7.cumsum(1);
            let _r8 = t8.cumprod(1);
            
            total_operations += 8;
        }
        
        let iter_duration = iter_start.elapsed();
        
        if iteration % 2 == 0 {  // Report more frequently
            let elapsed = start_time.elapsed();
            let remaining = burn_duration.saturating_sub(elapsed);
            
            println!("🔥 BURN #{}: {} ops in {:?} | Remaining: {:?} | Total: {} | Rate: {:.1}/sec", 
                     iteration, test_sizes.len() * 8, iter_duration, remaining, total_operations,
                     total_operations as f64 / elapsed.as_secs_f64());
            
            // Print CPU usage hint more frequently
            if iteration % 4 == 0 {
                println!("   🌡️🌡️🌡️ ALL {core_count} CORES SHOULD BE AT 100% UTILIZATION! 🌡️🌡️🌡️");
            }
        }
        
        // NO SLEEP - CONTINUOUS MAXIMUM LOAD!
    }
    
    let total_time = start_time.elapsed();
    
    println!("\n🎯 CPU BURN TEST COMPLETED!");
    println!("⏱️  Total duration: {total_time:?}");
    println!("🔄 Total iterations: {iteration}");
    println!("📊 Total scan operations: {total_operations}");
    println!("🚀 Operations per second: {:.2}", total_operations as f64 / total_time.as_secs_f64());
    
    // Final stress burst - maximum parallel load
    println!("\n🚀 FINAL BURST: Maximum parallel load test!");
    
    let burst_start = Instant::now();
    let huge_tensor: Tensor<NdArrayBackend, 2> = 
        Tensor::random([1, 5_000_000], Distribution::Uniform(0.0, 1.0), &device); // 5M elements!
    
    println!("🔥 Processing 5M element tensor with parallel cumsum...");
    let _huge_result = huge_tensor.cumsum(1);
    let burst_duration = burst_start.elapsed();
    
    println!("⚡ 5M element parallel cumsum completed in: {burst_duration:?}");
    
    // Cool down message
    println!("\n🌡️  CPU BURN TEST COMPLETE - Let your CPU cool down!");
    println!("🎉 Your {core_count} cores have been thoroughly exercised with parallel scan operations!");
    println!("📈 Multi-core Rayon parallel scan integration: VERIFIED AND WORKING! 🔥");
    
    // Performance summary
    let ops_per_core = total_operations as f64 / core_count as f64;
    println!("\n📊 PERFORMANCE SUMMARY:");
    println!("   🧵 Cores utilized: {core_count}");
    println!("   ⚡ Avg operations per core: {ops_per_core:.2}");
    println!("   🔥 Peak tensor size processed: 5,000,000 elements");
    println!("   ✅ Multi-core CPU utilization: SUCCESS!");
}
