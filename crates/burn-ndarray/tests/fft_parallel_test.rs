use burn_ndarray::NdArray;
use burn_tensor::{Tensor, TensorData, Shape};

#[test]
fn test_fft_parallel_threshold() {
    type TestBackend = NdArray;
    
    // Create a large array that should trigger parallel processing (>= 1024 elements)
    let size = 2048;
    let data: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
    let tensor_data = TensorData::new(data, Shape::new([size]));
    let tensor = Tensor::<TestBackend, 1>::from_data(tensor_data, &Default::default());
    
    println!("Large tensor shape: {:?}", tensor.dims());
    
    // Test FFT - should use parallel processing
    let fft_result = tensor.fft(0);
    println!("Large FFT result shape: {:?}", fft_result.dims());
    
    // Verify we get reasonable output
    assert_eq!(fft_result.dims(), [size]);
    println!("Large FFT completed successfully");
}

#[test]
fn test_fft_small_sequential() {
    type TestBackend = NdArray;
    
    // Create a small array that should use sequential processing (< 1024 elements)
    let size = 16;
    let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).cos()).collect();
    let tensor_data = TensorData::new(data, Shape::new([size]));
    let tensor = Tensor::<TestBackend, 1>::from_data(tensor_data, &Default::default());
    
    println!("Small tensor shape: {:?}", tensor.dims());
    
    // Test FFT - should use sequential processing  
    let fft_result = tensor.fft(0);
    println!("Small FFT result shape: {:?}", fft_result.dims());
    
    // Verify we get reasonable output
    assert_eq!(fft_result.dims(), [size]);
    println!("Small FFT completed successfully");
}
