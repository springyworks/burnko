// Add the necessary path for the dual shader backend
use burn::{
    backend::Autodiff,
    tensor::{backend::Backend, Tensor},
};
use burn_wgpu::{Wgpu, WgpuDevice};

// We need to add the path manually since it's not a published crate
#[path = "wgpu-direct-video/burn-wgpu-dual/src/lib.rs"]
mod burn_wgpu_dual;

use burn_wgpu_dual::DualShaderBackend;

type MyBackend = DualShaderBackend<Wgpu>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing DualShaderBackend...");
    
    // Create device
    let device = WgpuDevice::default();
    println!("Backend name: {}", MyBackend::name());
    
    // Test basic tensor operations
    let tensor1 = Tensor::<MyBackend, 2>::random([2, 3], burn::tensor::Distribution::Default, &device);
    let tensor2 = Tensor::<MyBackend, 2>::random([2, 3], burn::tensor::Distribution::Default, &device);
    
    println!("Created tensors with shape: {:?}", tensor1.shape());
    
    // Test addition
    let result = tensor1 + tensor2;
    println!("Addition result shape: {:?}", result.shape());
    
    // Test matrix multiplication  
    let a = Tensor::<MyBackend, 2>::random([3, 4], burn::tensor::Distribution::Default, &device);
    let b = Tensor::<MyBackend, 2>::random([4, 5], burn::tensor::Distribution::Default, &device);
    let matmul_result = a.matmul(b);
    println!("Matrix multiplication result shape: {:?}", matmul_result.shape());
    
    // Test more complex operations
    let tensor = Tensor::<MyBackend, 2>::random([10, 10], burn::tensor::Distribution::Default, &device);
    let mean = tensor.clone().mean();
    let sum = tensor.clone().sum();
    let exp_tensor = tensor.clone().exp();
    
    println!("Mean shape: {:?}", mean.shape());
    println!("Sum shape: {:?}", sum.shape());
    println!("Exp tensor shape: {:?}", exp_tensor.shape());
    
    // Test with autodiff
    type AutodiffBackend = Autodiff<MyBackend>;
    let grad_tensor = Tensor::<AutodiffBackend, 2>::random([5, 5], burn::tensor::Distribution::Default, &device);
    let grad_result = grad_tensor.clone().exp().sum().backward();
    println!("Gradient computation completed");
    
    println!("✅ DualShaderBackend test completed successfully!");
    Ok(())
}
