use burn_ndarray::NdArray;
use burn_wgpu::Wgpu;
use burn_tensor::{Tensor, TensorData, Shape, Distribution};

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
            
            max_diff = max_diff.max(diff);
            max_rel_error = max_rel_error.max(rel_error);
            
            // For analytical sequence [1,2,3,...,N], cumsum[i] = (i+1)*(i+2)/2
            let expected = ((idx + 1) * (idx + 2) / 2) as f32;
            
            println!("   Index {}: CPU={:.1}, GPU={:.1}, Expected={:.1}, Diff={:.2}, RelErr={:.2e}",
                     idx, cpu_val, gpu_val, expected, diff, rel_error);
        }
        
        // Final sum should be N*(N+1)/2
        let expected_sum = (size * (size + 1) / 2) as f32;
        let cpu_sum = cpu_values[size - 1];
        let gpu_sum = gpu_values[size - 1];
        
        println!("   📊 Final cumsum: Expected={:.1}, CPU={:.1}, GPU={:.1}", 
                 expected_sum, cpu_sum, gpu_sum);
        println!("   📏 Max absolute diff: {:.2}", max_diff);
        println!("   📐 Max relative error: {:.2e}", max_rel_error);
        
        // Determine if this is acceptable precision
        let acceptable_rel_error = 1e-5; // 0.001% relative error
        if max_rel_error <= acceptable_rel_error {
            println!("   ✅ PASS: Precision is acceptable");
        } else {
            println!("   ❌ FAIL: Precision exceeds tolerance");
            
            // Show where the differences start to become significant
            let mut first_significant_diff_idx = None;
            for i in 0..size {
                let cpu_val = cpu_values[i];
                let gpu_val = gpu_values[i];
                let diff = (cpu_val - gpu_val).abs();
                let rel_error = if cpu_val != 0.0 { diff / cpu_val.abs() } else { diff };
                
                if rel_error > acceptable_rel_error {
                    first_significant_diff_idx = Some(i);
                    break;
                }
            }
            
            if let Some(idx) = first_significant_diff_idx {
                println!("   🔍 First significant difference at index {}", idx);
            }
        }
    }
    
    println!("\n🎯 Precision test completed!");
    println!("\n💡 Recommendations:");
    println!("   - For large cumulative sums, consider using relative tolerance rather than absolute");
    println!("   - Use double precision (f64) for high-precision requirements");
    println!("   - For very large arrays, consider Kahan summation algorithm");
}
