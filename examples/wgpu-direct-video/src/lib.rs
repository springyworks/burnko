//! # Simple Burn Tensor Visualizer
//! 
//! Zero-copy tensor visualization with dual-purpose shaders.
//! 
//! ## Super Simple API
//! 
//! ```rust
//! use burn::tensor::{Tensor, Shape, Distribution, Device};
//! use burn::backend::Wgpu;
//! use wgpu_direct_video::visualize_tensor;
//! 
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let device = Device::<Wgpu>::default();
//!     let tensor = Tensor::<Wgpu, 2>::random(
//!         Shape::new([512, 512]), 
//!         Distribution::Normal(0.0, 1.0), 
//!         &device
//!     );
//!     
//!     // That's it! One function call to visualize any tensor
//!     visualize_tensor(&tensor).await?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Key Features
//! - 🔥 **Single shader** for both compute and graphics
//! - 🚀 **Zero-copy** - tensor operations stay on GPU
//! - 💫 **Real-time** matrix transformation visualization
//! - ⚡ **60+ FPS** performance

pub mod simple_visualizer;

// Re-export the simple API
pub use simple_visualizer::{visualize_tensor, Backend};
