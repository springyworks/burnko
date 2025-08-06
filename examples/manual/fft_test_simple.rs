use burn_tensor::{Tensor, backend::ndarray::NdArray, Device};

fn main() {
    let device = Device::<NdArray>::default();
    
    // Create a simple test tensor
    let tensor = Tensor::<NdArray, 1>::from_data([1.0, 0.0, -1.0, 0.0], &device);
    
    println!("Input tensor: {:?}", tensor.to_data());
    
    // Test FFT - this should currently just return the input (placeholder)
    let fft_result = tensor.clone().fft(0);
    println!("FFT result: {:?}", fft_result.to_data());
    
    // Test IFFT - this should also return the input (placeholder)
    let ifft_result = fft_result.ifft(0);
    println!("IFFT result: {:?}", ifft_result.to_data());
}
