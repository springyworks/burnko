use burn_ndarray::NdArray;
use burn_wgpu::Wgpu;
use burn::tensor::{Tensor, TensorData, Shape};

type NdArrayBackend = NdArray<f32>;
type WgpuBackend = Wgpu;

/// Simple test to compare cumsum between NdArray (CPU) and Wgpu (GPU) backends
/// This focuses on the tolerance issue you mentioned in large data
fn main() {
    let cpu_device = burn_ndarray::NdArrayDevice::Cpu;
    let gpu_device = Default::default();

    println!("🧮 Testing cumsum precision: NdArray vs Wgpu backends");
    println!("{}", "=".repeat(60));

    // Test different sizes to identify where precision issues start
    let test_sizes = vec![100, 1_000, 10_000, 100_000];
    
    for size in test_sizes {
        println!("\n📏 Testing size: {} elements", size);
        
        // Create analytical data [1, 2, 3, ..., N]
        let data: Vec<f32> = (1..=size).map(|i| i as f32).collect();
        
        // Create CPU tensor (NdArray)
        let cpu_tensor: Tensor<NdArrayBackend, 1> = Tensor::from_data(
            TensorData::new(data.clone(), Shape::new([size])),
            &cpu_device
        );
        
        // Create GPU tensor (Wgpu)
        let gpu_tensor: Tensor<WgpuBackend, 1> = Tensor::from_data(
            TensorData::new(data, Shape::new([size])),
            &gpu_device
        );
        
        // Compute cumsum on both backends
        let cpu_result = cpu_tensor.cumsum(0);
        let gpu_result = gpu_tensor.cumsum(0);
        
        // Get the results as vectors
        let cpu_values: Vec<f32> = cpu_result.to_data().to_vec().unwrap();
        let gpu_values: Vec<f32> = gpu_result.to_data().to_vec().unwrap();
        
        // Compare last few values (where errors accumulate most)
        let check_indices = if size >= 10 {
            vec![size - 3, size - 2, size - 1]
        } else {
            vec![size - 1]
        };
        
        let mut max_diff = 0.0f32;
        let mut max_rel_error = 0.0f32;
        
        for &idx in &check_indices {
            let cpu_val = cpu_values[idx];
            let gpu_val = gpu_values[idx];
            let diff = (cpu_val - gpu_val).abs();
            let rel_error = if cpu_val != 0.0 { diff / cpu_val.abs() } else { diff };
            
            if diff > max_diff {
                max_diff = diff;
            }
            if rel_error > max_rel_error {
                max_rel_error = rel_error;
            }
            
            println!("  Index {}: CPU={:.6}, GPU={:.6}, diff={:.2e}, rel_err={:.2e}%", 
                     idx, cpu_val, gpu_val, diff, rel_error * 100.0);
        }
        
        println!("  📊 Summary for {} elements:", size);
        println!("    Max absolute difference: {:.2e}", max_diff);
        println!("    Max relative error: {:.2e}%", max_rel_error * 100.0);
        
        // Check if results are "close enough"
        let tolerance = 1e-4; // Reasonable tolerance for f32
        if max_rel_error < tolerance {
            println!("    ✅ Results are within tolerance!");
        } else {
            println!("    ⚠️  Results exceed tolerance (>{:.2e}%)", tolerance * 100.0);
        }
    }
    
    println!("\n🏁 Test completed!");
}
