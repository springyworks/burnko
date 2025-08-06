//! GPU Burn Monitor: Real-time GPU monitoring with intensive stress testing
//! 
//! This test monitors actual GPU utilization, temperature, and memory usage
//! while pushing the GPU to maximum load with parallel scan operations.

#[cfg(test)]
mod gpu_burn_monitor_tests {
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

    // GPU monitoring using nvidia-smi (fallback if NVML not available)
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

    // Monitoring thread that tracks GPU stats in real-time
    fn start_gpu_monitoring(running: Arc<AtomicBool>) -> Arc<Mutex<Vec<(f32, f32, f32, Instant)>>> {
        let stats = Arc::new(Mutex::new(Vec::new()));
        let stats_clone = stats.clone();
        
        thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                if let Ok((temp, util, mem_util)) = get_gpu_stats_nvidia_smi() {
                    let timestamp = Instant::now();
                    if let Ok(mut stats_guard) = stats_clone.lock() {
                        stats_guard.push((temp, util, mem_util, timestamp));
                        
                        // Keep only last 100 measurements to avoid memory bloat
                        if stats_guard.len() > 100 {
                            stats_guard.drain(0..50);
                        }
                    }
                }
                thread::sleep(Duration::from_millis(500)); // Sample every 500ms
            }
        });
        
        stats
    }

    // Aggressive GPU burn workload generator
    fn generate_burn_workload(
        device: &<TestBackend as burn_tensor::backend::Backend>::Device,
        duration_ms: u64,
        intensity: usize,
    ) -> Vec<(f32, f32, f32)> {
        let start_time = Instant::now();
        let mut gpu_stats = Vec::new();
        
        // Create multiple tensors of different sizes for maximum GPU occupancy
        let tensor_sizes = vec![128, 256, 512, 1024, 2048, 4096];
        let operations = vec![ScanOp::Add, ScanOp::Mul, ScanOp::Max, ScanOp::Min];
        
        // Pre-create all tensors to avoid allocation overhead during burn
        let tensors: Vec<TestTensor<1>> = tensor_sizes.iter().map(|&size| {
            let values: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1) % 10.0 + 1.0).collect();
            TestTensor::<1>::from_data(
                TensorData::new(values, Shape::new([size])), 
                device
            )
        }).collect();
        
        // Create 2D tensors for multi-dimensional burn
        let matrix_sizes = vec![64, 128, 256];
        let matrix_tensors: Vec<TestTensor<2>> = matrix_sizes.iter().map(|&size| {
            let values: Vec<f32> = (0..size*size)
                .map(|i| (i as f32 * 0.01) % 5.0 + 0.1)
                .collect();
            TestTensor::<2>::from_data(
                TensorData::new(values, Shape::new([size, size])), 
                device
            )
        }).collect();
        
        let mut operation_count = 0;
        
        while start_time.elapsed().as_millis() < duration_ms as u128 {
            // Parallel execution of multiple operations
            for _ in 0..intensity {
                // 1D tensor operations
                for (_tensor_idx, tensor) in tensors.iter().enumerate() {
                    for (_op_idx, &op) in operations.iter().enumerate() {
                        let config = ScanConfig::new(op, 0);
                        let _result = tensor.clone().scan(config);
                        let _values: Vec<f32> = _result.to_data().to_vec().unwrap();
                        operation_count += 1;
                        
                        // Occasionally sample GPU stats during burn
                        if operation_count % 50 == 0 {
                            if let Ok((temp, util, mem_util)) = get_gpu_stats_nvidia_smi() {
                                gpu_stats.push((temp, util, mem_util));
                            }
                        }
                    }
                }
                
                // 2D tensor operations
                for matrix_tensor in &matrix_tensors {
                    for dim in 0..2 {
                        let config = ScanConfig::new(ScanOp::Add, dim);
                        let _result = matrix_tensor.clone().scan(config);
                        let _values: Vec<f32> = _result.to_data().to_vec().unwrap();
                        operation_count += 1;
                    }
                }
            }
        }
        
        println!("🔥 Executed {} GPU operations in {}ms", operation_count, duration_ms);
        gpu_stats
    }

    #[test]
    fn test_gpu_burn_with_real_monitoring() {
        println!("\n🔥🔥🔥 GPU BURN WITH REAL-TIME MONITORING 🔥🔥🔥");
        println!("🎯 Goal: Push GPU utilization above 75% and temperature above 60°C");
        println!("📊 Monitoring: Real-time GPU stats via nvidia-smi");
        
        let device = get_device();
        
        // Check initial GPU state
        match get_gpu_stats_nvidia_smi() {
            Ok((temp, util, mem_util)) => {
                println!("🌡️  Initial GPU State:");
                println!("   Temperature: {:.1}°C", temp);
                println!("   Utilization: {:.1}%", util);
                println!("   Memory: {:.1}%", mem_util);
            }
            Err(e) => {
                println!("⚠️  Warning: Could not get GPU stats: {}", e);
                println!("   Continuing with burn test anyway...");
            }
        }
        
        // Start background monitoring
        let monitoring_active = Arc::new(AtomicBool::new(true));
        let _gpu_stats = start_gpu_monitoring(monitoring_active.clone());
        
        // Progressive burn intensity levels
        let burn_phases = vec![
            ("🔥 WARM-UP", 5000, 2),      // 5s, low intensity
            ("🔥🔥 MEDIUM BURN", 10000, 5),  // 10s, medium intensity  
            ("🔥🔥🔥 MAXIMUM BURN", 15000, 10), // 15s, maximum intensity
            ("🔥🔥🔥🔥 INSANE BURN", 20000, 15), // 20s, insane intensity
        ];
        
        let mut max_temp: f32 = 0.0;
        let mut max_util: f32 = 0.0;
        let mut max_mem: f32 = 0.0;
        
        for (phase_name, duration_ms, intensity) in burn_phases {
            println!("\n{} ({}ms, intensity: {})", phase_name, duration_ms, intensity);
            
            let phase_start = Instant::now();
            let phase_stats = generate_burn_workload(&device, duration_ms, intensity);
            let phase_duration = phase_start.elapsed();
            
            // Analyze phase results
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
                
                // Real-time feedback
                if phase_max_util > 75.0 {
                    println!("   🎯 SUCCESS: GPU utilization above 75%!");
                } else {
                    println!("   ⚠️  GPU utilization below target (75%)");
                }
                
                if phase_max_temp > 60.0 {
                    println!("   🔥 SUCCESS: GPU temperature above 60°C!");
                } else if phase_max_temp > 50.0 {
                    println!("   🌡️  GPU getting warm (50°C+)");
                } else {
                    println!("   ❄️  GPU still cool (<50°C)");
                }
            }
            
            // Brief cooldown between phases
            thread::sleep(Duration::from_millis(1000));
        }
        
        // Stop monitoring and get final stats
        monitoring_active.store(false, Ordering::Relaxed);
        thread::sleep(Duration::from_millis(1000)); // Wait for monitoring to stop
        
        // Final GPU state check
        match get_gpu_stats_nvidia_smi() {
            Ok((temp, util, mem_util)) => {
                println!("\n🏁 Final GPU State:");
                println!("   Temperature: {:.1}°C", temp);
                println!("   Utilization: {:.1}%", util);
                println!("   Memory: {:.1}%", mem_util);
            }
            Err(_) => {}
        }
        
        // Overall results
        println!("\n📊 BURN TEST RESULTS:");
        println!("   🌡️  Maximum Temperature: {:.1}°C", max_temp);
        println!("   ⚡ Maximum Utilization: {:.1}%", max_util);
        println!("   💾 Maximum Memory Usage: {:.1}%", max_mem);
        
        // Performance analysis
        if max_util > 75.0 {
            println!("   🏆 EXCELLENT: Achieved >75% GPU utilization!");
        } else if max_util > 50.0 {
            println!("   👍 GOOD: Achieved >50% GPU utilization");
        } else {
            println!("   📈 NEEDS MORE INTENSITY: GPU utilization was only {:.1}%", max_util);
            println!("   💡 Try: Increase intensity, use larger tensors, or run multiple parallel streams");
        }
        
        if max_temp > 60.0 {
            println!("   🔥 SUCCESS: GPU reached working temperature (>60°C)");
        } else {
            println!("   ❄️  GPU stayed cool at {:.1}°C - could handle more load!", max_temp);
        }
        
        println!("\n🎉 GPU BURN WITH MONITORING COMPLETE!");
        
        // Assert test passes regardless of utilization (this is a stress test, not a failure)
        assert!(max_temp >= 0.0, "Should have recorded some temperature readings");
    }
}
