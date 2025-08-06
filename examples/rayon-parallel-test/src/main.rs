use burn_ndarray::{NdArray, NdArrayDevice};
use burn::tensor::Tensor;
use std::time::Instant;

type Backend = NdArray<f32>;

fn main() {
    // Set number of threads explicitly
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()
        .unwrap();
    
    println!("Testing with {} threads", rayon::current_num_threads());
    
    // Create a large tensor
    let device = NdArrayDevice::Cpu;
    let tensor: Tensor<Backend, 1> = Tensor::arange(0..1_000_000, &device).float();
    
    println!("Testing cumsum on {} elements", tensor.dims()[0]);
    
    let start = Instant::now();
    let result = tensor.cumsum(0);
    let duration = start.elapsed();
    
    println!("Cumsum completed in {:?}", duration);
    
    // Verify correctness on first few elements - convert to int for comparison
    let first_10_data = result.slice([0..10]).to_data();
    let first_10_f32: Vec<f32> = first_10_data.to_vec().unwrap();
    let first_10: Vec<i32> = first_10_f32.iter().map(|&x| x as i32).collect();
    println!("First 10 results: {:?}", first_10);
    
    // Expected: [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]
    let expected = vec![0, 1, 3, 6, 10, 15, 21, 28, 36, 45];
    println!("Expected:         {:?}", expected);
    
    if first_10 == expected {
        println!("✓ Results are correct!");
    } else {
        println!("✗ Results are incorrect!");
        println!("Difference: {:?}", first_10.iter().zip(&expected).map(|(a, b)| a - b).collect::<Vec<_>>());
    }
}
