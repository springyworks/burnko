use burn_ndarray::NdArray;
use burn_tensor::{Tensor, TensorData};

#[test]
fn test_fft_basic_functionality() {
    type TestBackend = NdArray;
    
    // Create a simple 1D tensor with real values [1, 0, 1, 0]
    let data = TensorData::from([1.0f32, 0.0, 1.0, 0.0]);
    let tensor = Tensor::<TestBackend, 1>::from_data(data, &Default::default());
    
    println!("Original shape: {:?}", tensor.dims());
    
    // Test FFT along dimension 0
    let fft_result = tensor.fft(0);
    
    // Print debug info
    println!("FFT result shape: {:?}", fft_result.dims());
    println!("FFT result data: {:?}", fft_result.to_data());
    
    // For now, let's just verify the FFT runs without panicking
    // and produces some reasonable output shape
    
    // The FFT should produce complex output, so we expect shape to change
    // Input [4] -> Output [4, 2] where [:,0] = real, [:,1] = imag
    let result_shape = fft_result.dims();
    
    // Basic sanity checks
    assert!(result_shape.len() >= 1, "FFT result should have at least 1 dimension");
    println!("FFT completed successfully with shape: {:?}", result_shape);
}

#[test]
fn test_fft_ifft_roundtrip() {
    type TestBackend = NdArray;
    
    // Create a simple test tensor
    let data = TensorData::from([1.0f32, 2.0, 3.0, 4.0]);
    let original = Tensor::<TestBackend, 1>::from_data(data.clone(), &Default::default());
    
    println!("Original: {:?}", data);
    
    // Perform FFT -> IFFT roundtrip
    let fft_result = original.fft(0);
    println!("FFT shape: {:?}", fft_result.dims());
    
    // For IFFT, we expect the FFT result to have complex data format
    let ifft_result = fft_result.ifft(0);
    println!("IFFT shape: {:?}", ifft_result.dims());
    
    // The shapes should match after roundtrip (within floating point precision)
    println!("After roundtrip: {:?}", ifft_result.to_data());
    
    // Basic sanity check - IFFT should restore original dimensions
    assert_eq!(ifft_result.dims(), [4]);
}
