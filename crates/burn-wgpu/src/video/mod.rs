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
pub use pipeline::*;
pub use stream::*;
pub use tensor_texture::*;

use crate::{CubeBackend, WgpuRuntime};
use burn_tensor::Tensor;

/// Main trait for tensor video streaming capabilities
pub trait TensorVideoStream<B: burn_tensor::backend::Backend> {
    /// Create a new video stream with the specified configuration
    fn create_stream(&self, config: VideoConfig) -> VideoStream;
    
    /// Push a tensor frame to the video stream
    fn push_frame(&mut self, tensor: Tensor<B, 2>);
    
    /// Set the colormap for visualization
    fn set_colormap(&mut self, colormap: ColorMap);
}

/// Implement video streaming for WGPU backend tensors
impl<F, I, BT> TensorVideoStream<CubeBackend<WgpuRuntime, F, I, BT>> 
    for Tensor<CubeBackend<WgpuRuntime, F, I, BT>, 2>
where
    F: crate::FloatElement,
    I: crate::IntElement,
    BT: crate::BoolElement,
{
    fn create_stream(&self, config: VideoConfig) -> VideoStream {
        VideoStream::new(config)
    }
    
    fn push_frame(&mut self, tensor: Tensor<CubeBackend<WgpuRuntime, F, I, BT>, 2>) {
        // Implementation will convert tensor to texture and push to stream
        todo!("Implement tensor to texture conversion and streaming")
    }
    
    fn set_colormap(&mut self, colormap: ColorMap) {
        // Implementation will update the visualization colormap
        todo!("Implement colormap updates")
    }
}
