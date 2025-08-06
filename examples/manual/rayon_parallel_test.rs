use burn_ndarray::NdArray;
use burn_tensor::{Tensor, Device};
use std::time::Instant;

fn main() {
    // Set number of threads explicitly
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()
        .unwrap();
    
    println!("Testing with {} threads", rayon::current_num_threads());
    
    // Create a large tensor
    let device = Device::Cpu;
    let tensor: Tensor<NdArray, 1> = Tensor::arange(0..1_000_000, &device);
    
    println!("Testing cumsum on {} elements", tensor.dims()[0]);
    
    let start = Instant::now();
    let result = tensor.cumsum(0);
    let duration = start.elapsed();
    
    println!("Cumsum completed in {:?}", duration);
    
    // Verify correctness on first few elements
    let first_10 = result.slice([0..10]).into_data().convert::<i32>().value;
    println!("First 10 results: {:?}", first_10);
    
    // Expected: [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]
    let expected = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45];
    println!("Expected:         {:?}", expected);
    
    if first_10 == expected {
        println!("✓ Results are correct!");
    } else {
        println!("✗ Results are incorrect!");
    }
}
