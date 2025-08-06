//! # Super Simple Tensor Visualizer Demo
//! 
//! Just one function call - that's all you need!

use burn::{
    tensor::{Tensor, Shape, Distribution, Device},
    backend::Wgpu,
};
use wgpu_direct_video::visualize_tensor;

type Backend = Wgpu<f32>;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Super Simple Burn Tensor Visualizer");
    println!("=====================================");
    
    // Step 1: Create a tensor (any tensor will do!)
    let device = Device::<Backend>::default();
    let tensor = Tensor::<Backend, 2>::random(
        Shape::new([512, 512]), 
        Distribution::Normal(0.0, 1.0), 
        &device
    );
    
    println!("✨ Created 512x512 random tensor");
    
    // Step 2: Visualize it - that's it!
    println!("🚀 Starting dual-shader visualization...");
    visualize_tensor(&tensor).await?;
    
    println!("🎉 Visualization complete!");
    
    Ok(())
}
