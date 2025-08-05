//! WGPU Direct Video Pipeline
//! 
//! This module provides GPU-native real-time tensor visualization capabilities.
//! Tensors are streamed directly from GPU compute operations to graphics rendering
//! without CPU roundtrips, enabling unprecedented performance for live tensor monitoring.

pub mod config;
pub mod pipeline;
pub mod stream;
pub mod tensor_texture;

pub use config::*;
pub use tensor_texture::*;
pub use pipeline::*;
// pub use stream::*;  // Temporarily commented out

use burn_tensor::backend::Backend;
use burn_tensor::Tensor;

/// Trait for video streaming backends (placeholder)
pub trait VideoBackend {
    // fn create_stream(&self, config: VideoConfig) -> VideoStream;  // Temporarily commented out
}
