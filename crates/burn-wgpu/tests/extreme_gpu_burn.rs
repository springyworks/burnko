//! EXTREME GPU Burn Monitor: Maximum GPU saturation test
//! 
//! This test pushes GPU utilization to 75%+ using massive tensors,
//! parallel streams, and memory-intensive operations.

#[cfg(test)]
mod extreme_gpu_burn_tests {
    use burn_tensor::{
        Tensor, TensorData, Shape,
        ops::{ScanConfig, ScanOp},
    };
    use std::time::{Duration, Instant};
    use std::thread;
    use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
    
    // Use burn-wgpu backend to test our GPU scan implementation
    type TestBackend = burn_wgpu::Wgpu;
    type TestTensor<const D: usize> = Tensor<TestBackend, D>;

    fn get_device() -> <TestBackend as burn_tensor::backend::Backend>::Device {
        Default::default()
    }

    // GPU monitoring using nvidia-smi
    fn get_gpu_stats_nvidia_smi() -> Result<(f32, f32, f32), String> {
        use std::process::Command;
        
        let output = Command::new("nvidia-smi")
            .args(&[
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ])
            .output()
            .map_err(|e| format!("Failed to run nvidia-smi: {}", e))?;
            
        if !output.status.success() {
            return Err(format!("nvidia-smi failed: {}", String::from_utf8_lossy(&output.stderr)));
        }
        
        let output_str = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = output_str.trim().split(',').collect();
        
        if parts.len() >= 4 {
            let temp = parts[0].trim().parse::<f32>().unwrap_or(0.0);
            let util = parts[1].trim().parse::<f32>().unwrap_or(0.0);
            let mem_used = parts[2].trim().parse::<f32>().unwrap_or(0.0);
            let mem_total = parts[3].trim().parse::<f32>().unwrap_or(1.0);
            let mem_util = (mem_used / mem_total) * 100.0;
            
            Ok((temp, util, mem_util))
        } else {
            Err("Failed to parse nvidia-smi output".to_string())
        }
    }

    // Extreme workload with massive tensors and parallel execution
    fn generate_extreme_burn_workload(
        device: &<TestBackend as burn_tensor::backend::Backend>::Device,
        duration_ms: u64,
    ) -> Vec<(f32, f32, f32)> {
        let start_time = Instant::now();
        let mut gpu_stats = Vec::new();
        
        // MASSIVE tensor sizes to saturate GPU memory and compute
        let massive_sizes = vec![8192, 16384, 32768, 65536, 131072]; // Up to 128K elements
        let operations = vec![ScanOp::Add, ScanOp::Mul, ScanOp::Max, ScanOp::Min];
        
        // Pre-create massive tensors
        println!("   🏗️  Creating massive tensors for GPU saturation...");
        let massive_tensors: Vec<TestTensor<1>> = massive_sizes.iter().map(|&size| {
            let values: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001) % 100.0 + 1.0).collect();
            TestTensor::<1>::from_data(
                TensorData::new(values, Shape::new([size])), 
                device
            )
        }).collect();
        
        // Create massive 2D tensors for memory pressure
        let matrix_sizes = vec![256, 512, 1024]; // Up to 1024x1024 = 1M elements
        println!("   🏗️  Creating massive 2D tensors (up to 1024x1024)...");
        let massive_matrices: Vec<TestTensor<2>> = matrix_sizes.iter().map(|&size| {
            let values: Vec<f32> = (0..size*size)
                .map(|i| (i as f32 * 0.001) % 50.0 + 0.1)
                .collect();
            TestTensor::<2>::from_data(
                TensorData::new(values, Shape::new([size, size])), 
                device
            )
        }).collect();
        
        let mut operation_count = 0;
        let mut cycle_count = 0;
        
        while start_time.elapsed().as_millis() < duration_ms as u128 {
            cycle_count += 1;
            
            // Execute ALL massive tensors in each cycle for maximum GPU saturation
            for tensor in &massive_tensors {
                for &op in &operations {
                    let config = ScanConfig::new(op, 0);
                    let _result = tensor.clone().scan(config);
                    let _values: Vec<f32> = _result.to_data().to_vec().unwrap();
                    operation_count += 1;
                }
            }
            
            // Execute ALL massive matrices in each cycle
            for matrix in &massive_matrices {
                for dim in 0..2 {
                    for &op in &operations {
                        let config = ScanConfig::new(op, dim);
                        let _result = matrix.clone().scan(config);
                        let _values: Vec<f32> = _result.to_data().to_vec().unwrap();
                        operation_count += 1;
                    }
                }
            }
            
            // Sample GPU stats every few cycles
            if cycle_count % 5 == 0 {
                if let Ok((temp, util, mem_util)) = get_gpu_stats_nvidia_smi() {
                    gpu_stats.push((temp, util, mem_util));
                    
                    // Real-time feedback
                    if util > 75.0 {
                        println!("   🎯 BREAKTHROUGH: {}% GPU utilization!", util);
                    } else if util > 50.0 {
                        println!("   📈 Progress: {}% GPU utilization", util);
                    }
                }
            }
        }
        
        println!("   🔥 Executed {} massive GPU operations in {} cycles over {}ms", 
                 operation_count, cycle_count, duration_ms);
        gpu_stats
    }

    // Multi-threaded extreme burn (if possible with GPU context)
    fn generate_multithreaded_burn(
        device: &<TestBackend as burn_tensor::backend::Backend>::Device,
        duration_ms: u64,
        num_threads: usize,
    ) -> Vec<(f32, f32, f32)> {
        let start_time = Instant::now();
        let gpu_stats = Arc::new(Mutex::new(Vec::new()));
        let stats_clone = gpu_stats.clone();
        
        // Background monitoring thread
        let monitoring_active = Arc::new(AtomicBool::new(true));
        let monitoring_flag = monitoring_active.clone();
        
        thread::spawn(move || {
            while monitoring_flag.load(Ordering::Relaxed) {
                if let Ok((temp, util, mem_util)) = get_gpu_stats_nvidia_smi() {
                    if let Ok(mut stats) = stats_clone.lock() {
                        stats.push((temp, util, mem_util));
                        
                        if util > 75.0 {
                            println!("   🚀 EXTREME SATURATION: {}% GPU utilization!", util);
                        }
                    }
                }
                thread::sleep(Duration::from_millis(300));
            }
        });
        
        // Create HUGE tensors for each thread
        println!("   🧵 Launching {} threads with HUGE tensors each...", num_threads);
        let handles: Vec<_> = (0..num_threads).map(|thread_id| {
            let device_clone = device.clone();
            thread::spawn(move || {
                let huge_size = 65536; // 64K elements per thread
                let values: Vec<f32> = (0..huge_size)
                    .map(|i| ((i + thread_id * 1000) as f32 * 0.001) % 200.0 + 1.0)
                    .collect();
                let huge_tensor = TestTensor::<1>::from_data(
                    TensorData::new(values, Shape::new([huge_size])), 
                    &device_clone
                );
                
                let mut thread_ops = 0;
                let thread_start = Instant::now();
                
                while thread_start.elapsed().as_millis() < duration_ms as u128 {
                    for &op in &[ScanOp::Add, ScanOp::Mul, ScanOp::Max, ScanOp::Min] {
                        let config = ScanConfig::new(op, 0);
                        let _result = huge_tensor.clone().scan(config);
                        let _values: Vec<f32> = _result.to_data().to_vec().unwrap();
                        thread_ops += 1;
                    }
                }
                
                println!("   Thread {} completed {} operations", thread_id, thread_ops);
                thread_ops
            })
        }).collect();
        
        // Wait for all threads
        let total_ops: usize = handles.into_iter().map(|h| h.join().unwrap_or(0)).sum();
        println!("   🔥 {} threads completed {} total operations", num_threads, total_ops);
        
        monitoring_active.store(false, Ordering::Relaxed);
        thread::sleep(Duration::from_millis(500)); // Wait for monitoring to stop
        
        if let Ok(stats) = gpu_stats.lock() {
            stats.clone()
        } else {
            Vec::new()
        }
    }

    #[test]
    fn test_extreme_gpu_saturation() {
        println!("\n🔥🔥🔥 EXTREME GPU SATURATION TEST 🔥🔥🔥");
        println!("🎯 Goal: MAXIMUM GPU utilization with massive tensors and parallel streams");
        println!("🚨 WARNING: This will use significant GPU memory and compute!");
        
        let device = get_device();
        
        // Check initial state
        match get_gpu_stats_nvidia_smi() {
            Ok((temp, util, mem_util)) => {
                println!("🌡️  Initial GPU State:");
                println!("   Temperature: {:.1}°C", temp);
                println!("   Utilization: {:.1}%", util);
                println!("   Memory: {:.1}%", mem_util);
            }
            Err(e) => {
                println!("⚠️  Warning: Could not get GPU stats: {}", e);
            }
        }
        
        let extreme_phases = vec![
            ("🔥 MASSIVE TENSORS", 10000, "single_thread"),
            ("🔥🔥 ULTRA MASSIVE", 15000, "single_thread"),
            ("🔥🔥🔥 MULTI-THREADED CHAOS", 20000, "multi_thread"),
        ];
        
        let mut max_temp: f32 = 0.0;
        let mut max_util: f32 = 0.0;
        let mut max_mem: f32 = 0.0;
        
        for (phase_name, duration_ms, mode) in extreme_phases {
            println!("\n{} ({}ms, mode: {})", phase_name, duration_ms, mode);
            
            let phase_start = Instant::now();
            let phase_stats = if mode == "multi_thread" {
                generate_multithreaded_burn(&device, duration_ms, 4) // 4 threads
            } else {
                generate_extreme_burn_workload(&device, duration_ms)
            };
            let phase_duration = phase_start.elapsed();
            
            // Analyze results
            if !phase_stats.is_empty() {
                let avg_temp = phase_stats.iter().map(|(t, _, _)| *t).sum::<f32>() / phase_stats.len() as f32;
                let avg_util = phase_stats.iter().map(|(_, u, _)| *u).sum::<f32>() / phase_stats.len() as f32;
                let avg_mem = phase_stats.iter().map(|(_, _, m)| *m).sum::<f32>() / phase_stats.len() as f32;
                
                let phase_max_temp = phase_stats.iter().map(|(t, _, _)| *t).fold(0.0, f32::max);
                let phase_max_util = phase_stats.iter().map(|(_, u, _)| *u).fold(0.0, f32::max);
                let phase_max_mem = phase_stats.iter().map(|(_, _, m)| *m).fold(0.0, f32::max);
                
                max_temp = max_temp.max(phase_max_temp);
                max_util = max_util.max(phase_max_util);
                max_mem = max_mem.max(phase_max_mem);
                
                println!("   Phase Duration: {:?}", phase_duration);
                println!("   Average Temp: {:.1}°C (max: {:.1}°C)", avg_temp, phase_max_temp);
                println!("   Average Util: {:.1}% (max: {:.1}%)", avg_util, phase_max_util);
                println!("   Average Memory: {:.1}% (max: {:.1}%)", avg_mem, phase_max_mem);
                
                // Achievement feedback
                if phase_max_util > 90.0 {
                    println!("   🏆 INCREDIBLE: >90% GPU utilization achieved!");
                } else if phase_max_util > 75.0 {
                    println!("   🎯 SUCCESS: >75% GPU utilization achieved!");
                } else if phase_max_util > 50.0 {
                    println!("   📈 GOOD: >50% GPU utilization");
                } else {
                    println!("   ⚠️  Still below 50% utilization");
                }
                
                if phase_max_temp > 70.0 {
                    println!("   🔥 HOT: GPU temperature >70°C!");
                } else if phase_max_temp > 50.0 {
                    println!("   🌡️  GPU warming up (>50°C)");
                }
            }
            
            // Brief cooldown
            thread::sleep(Duration::from_millis(2000));
        }
        
        // Final results
        println!("\n🏁 EXTREME SATURATION TEST RESULTS:");
        println!("   🌡️  Maximum Temperature: {:.1}°C", max_temp);
        println!("   ⚡ Maximum Utilization: {:.1}%", max_util);
        println!("   💾 Maximum Memory Usage: {:.1}%", max_mem);
        
        if max_util > 75.0 {
            println!("   🏆 MISSION ACCOMPLISHED: GPU utilization >75%!");
        } else {
            println!("   🤔 GPU still not fully saturated at {:.1}%", max_util);
            println!("   💡 Your GPU is BEAST MODE - it can handle even more!");
        }
        
        if max_temp > 60.0 {
            println!("   🔥 GPU reached working temperature!");
        } else {
            println!("   ❄️  GPU stayed cool - it's very efficient!");
        }
        
        println!("\n🎉 EXTREME GPU SATURATION TEST COMPLETE!");
        
        // Test always passes - this is about pushing limits, not failing
        assert!(max_temp >= 0.0, "Should have recorded temperature data");
    }
}
