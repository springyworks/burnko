use crate::tensor::CubeTensor;
use cubecl::server::Binding;

/// Trait for accessing underlying GPU buffers from CubeCL tensors
/// This enables zero-copy operations for advanced use cases
pub trait GpuBufferAccess<R: crate::CubeRuntime> {
    /// Get the binding for this tensor's buffer
    /// This is the safe way to access buffer information
    fn buffer_binding(&self) -> Binding;
}

impl<R: crate::CubeRuntime> GpuBufferAccess<R> for CubeTensor<R> {
    fn buffer_binding(&self) -> Binding {
        self.handle.clone().binding()
    }
}

#[cfg(feature = "wgpu")]
/// WGPU-specific buffer access implementation
/// This trait can be implemented by consumers who need direct WGPU buffer access
pub trait WgpuBufferAccess {
    /// Get direct access to the underlying WGPU buffer.
    /// This enables zero-copy GPU operations for advanced use cases like direct rendering.
    /// 
    /// # Safety
    /// This method provides direct access to the underlying GPU buffer. The caller must ensure:
    /// - The buffer is not modified while being used elsewhere
    /// - Proper synchronization if accessing from multiple contexts
    /// - The buffer remains valid for the lifetime of the returned reference
    /// 
    /// # Returns
    /// Returns the underlying `wgpu::Buffer` if available, or `None` if the buffer
    /// cannot be accessed.
    fn as_wgpu_buffer(&self) -> Option<&wgpu::Buffer>;
    
    /// Get access to the WGPU buffer with explicit client access
    /// This is a more explicit version that requires the client context
    fn get_wgpu_buffer_with_client(&self) -> Option<&wgpu::Buffer>;
}
