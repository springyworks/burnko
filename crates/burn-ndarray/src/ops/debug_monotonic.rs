//! Debug test for monotonic property violations in parallel scan
use crate::NdArray;
use burn_tensor::{Tensor, TensorData, Shape, Distribution};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_debug_monotonic_failure() {
        let device = Default::default();
        
        // Create a small random tensor to debug the monotonic issue
        let tensor: Tensor<NdArray<f32>, 1> = Tensor::random(
            [10], Distribution::Uniform(0.0, 1.0), &device
        );
        
        println!("Original tensor: {:?}", tensor.to_data().to_vec::<f32>().unwrap());
        
        let result = tensor.cumsum(0);
        let values: Vec<f32> = result.to_data().to_vec::<f32>().unwrap();
        
        println!("Cumsum result: {:?}", values);
        
        // Check monotonic property step by step
        for i in 1..values.len() {
            println!("Index {}: values[{}] = {:.6}, values[{}] = {:.6}, diff = {:.6}", 
                     i, i-1, values[i-1], i, values[i], values[i] - values[i-1]);
            
            if values[i] < values[i-1] {
                panic!("MONOTONIC VIOLATION at index {}: {} < {}", i, values[i], values[i-1]);
            }
        }
        
        println!("Monotonic property check passed for this tensor");
    }
    
    #[test]
    fn test_debug_known_values() {
        let device = Default::default();
        
        // Use known values to see what happens
        let known_data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let tensor: Tensor<NdArray<f32>, 1> = Tensor::from_data(
            TensorData::new(known_data.clone(), Shape::new([5])), &device
        );
        
        println!("Known input: {:?}", known_data);
        
        let result = tensor.cumsum(0);
        let values: Vec<f32> = result.to_data().to_vec::<f32>().unwrap();
        
        println!("Cumsum result: {:?}", values);
        
        // Expected: [0.1, 0.3, 0.6, 1.0, 1.5]
        let expected = vec![0.1, 0.3, 0.6, 1.0, 1.5];
        println!("Expected: {:?}", expected);
        
        for (i, (&actual, &expected)) in values.iter().zip(expected.iter()).enumerate() {
            println!("Index {}: actual = {:.6}, expected = {:.6}, diff = {:.6}", 
                     i, actual, expected, (actual - expected).abs());
        }
    }
}
