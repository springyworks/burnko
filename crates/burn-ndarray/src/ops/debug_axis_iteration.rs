//! Debug the axis iteration behavior to understand the parallel issue
use crate::NdArray;
use burn_tensor::{Tensor, TensorData, Shape};
use ndarray::Axis;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_debug_axis_iteration() {
        let device = Default::default();
        
        // Create a small 1D tensor
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor: Tensor<NdArray<f32>, 1> = Tensor::from_data(
            TensorData::new(data.clone(), Shape::new([5])), &device
        );
        
        let tensor_primitive = tensor.into_primitive();
        let array = &tensor_primitive.array;
        println!("Original array: {:?}", array);
        println!("Array shape: {:?}", array.shape());
        
        // Check what axis_iter_mut(Axis(0)) gives us for a 1D array
        let axis = Axis(0);
        let mut count = 0;
        for (i, lane) in array.axis_iter(axis).enumerate() {
            println!("Lane {}: shape={:?}, ndim={}", i, lane.shape(), lane.ndim());
            
            match lane.as_slice() {
                Some(slice) => {
                    println!("  Lane {} is a slice with {} elements: {:?}", i, slice.len(), slice);
                }
                None => {
                    println!("  Lane {} is NOT a slice (not contiguous)", i);
                }
            }
            
            count += 1;
            if count > 10 { break; } // Prevent infinite output
        }
        
        println!("Total lanes: {}", count);
    }
    
    #[test] 
    fn test_debug_2d_axis_iteration() {
        let device = Default::default();
        
        // Create a 2D tensor (3x4)
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let tensor: Tensor<NdArray<f32>, 2> = Tensor::from_data(
            TensorData::new(data, Shape::new([3, 4])), &device
        );
        
        let tensor_primitive = tensor.into_primitive();
        let array = &tensor_primitive.array;
        println!("2D array: {:?}", array);
        println!("2D shape: {:?}", array.shape());
        
        // Check axis_iter_mut(Axis(1)) for rows
        println!("\nIterating along axis 1 (rows):");
        for (i, lane) in array.axis_iter(Axis(1)).enumerate() {
            println!("Row {}: shape={:?}", i, lane.shape());
            if let Some(slice) = lane.as_slice() {
                println!("  Row {} slice: {:?}", i, slice);
            }
        }
        
        // Check axis_iter_mut(Axis(0)) for columns  
        println!("\nIterating along axis 0 (columns):");
        for (i, lane) in array.axis_iter(Axis(0)).enumerate() {
            println!("Col {}: shape={:?}", i, lane.shape());
            if let Some(slice) = lane.as_slice() {
                println!("  Col {} slice: {:?}", i, slice);
            }
        }
    }
}
