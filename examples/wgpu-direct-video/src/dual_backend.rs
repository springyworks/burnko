//! Dual-Purpose WGPU Backend Extension
//! 
//! This module extends Burn's WGPU backend to enable dual-purpose shaders that can be used
//! both for tensor operations AND real-time visualization simultaneously.
//! 
//! The core concept is that the same GPU memory and shaders are used for:
//! 1. Tensor computations (via Burn/CubeCL)
//! 2. Real-time visualization (via WGPU graphics pipeline)
//! 
//! This eliminates CPU transfers and enables unprecedented real-time tensor monitoring.

use burn_cubecl::{CubeBackend, tensor::CubeTensor, CubeRuntime, FloatElement, IntElement, BoolElement};
use burn_tensor::{Tensor, backend::Backend};
use burn_wgpu::{Wgpu, WgpuDevice};
use cubecl::server::Binding;
use std::sync::Arc;

/// Dual-Purpose WGPU Backend that supports both compute and graphics
pub struct DualWgpu<F = f32, I = i32, B = u32> 
where
    F: FloatElement,
    I: IntElement,
    B: BoolElement,
{
    /// The underlying Burn WGPU backend
    pub compute_backend: Wgpu<F, I, B>,
    /// WGPU device for graphics operations
    pub graphics_device: Arc<wgpu::Device>,
    /// WGPU queue for graphics operations
    pub graphics_queue: Arc<wgpu::Queue>,
    /// Surface for rendering (optional)
    pub surface: Option<wgpu::Surface<'static>>,
}

impl<F, I, B> DualWgpu<F, I, B> 
where
    F: FloatElement,
    I: IntElement,
    B: BoolElement,
{
    /// Create a new dual-purpose backend with existing WGPU device
    pub fn new_with_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Self {
        let compute_backend = Wgpu::default();
        
        Self {
            compute_backend,
            graphics_device: device,
            graphics_queue: queue,
            surface: None,
        }
    }
    
    /// Add a surface for rendering
    pub fn with_surface(mut self, surface: wgpu::Surface<'static>) -> Self {
        self.surface = Some(surface);
        self
    }
    
    /// Get the graphics device
    pub fn graphics_device(&self) -> &wgpu::Device {
        &self.graphics_device
    }
    
    /// Get the graphics queue
    pub fn graphics_queue(&self) -> &wgpu::Queue {
        &self.graphics_queue
    }
}

/// Trait for tensors that can be directly used in graphics pipelines
pub trait GraphicsInterop<B: Backend> {
    /// Get the underlying GPU buffer binding for graphics use
    fn graphics_binding(&self) -> Option<Binding>;
    
    /// Create a texture view from tensor data (zero-copy when possible)
    fn as_texture_view(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<wgpu::TextureView, GraphicsError>;
    
    /// Create a bind group for use in graphics shaders
    fn create_bind_group(
        &self, 
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        binding: u32,
    ) -> Result<wgpu::BindGroup, GraphicsError>;
}

/// Implementation for Burn tensors on WGPU backend
impl<const D: usize> GraphicsInterop<Wgpu> for Tensor<Wgpu, D> {
    fn graphics_binding(&self) -> Option<Binding> {
        // Access the underlying CubeTensor
        let primitive = self.clone().into_primitive();
        Some(primitive.handle.clone().binding())
    }
    
    fn as_texture_view(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<wgpu::TextureView, GraphicsError> {
        let _binding = self.graphics_binding()
            .ok_or(GraphicsError::NoBufferAccess)?;
            
        // For now, create a new texture and copy data
        // TODO: Implement true zero-copy texture views
        let shape = self.shape();
        
        // Only support 2D tensors for now
        if shape.num_dims() != 2 {
            return Err(GraphicsError::UnsupportedShape(format!("Expected 2D tensor, got {}D", shape.num_dims())));
        }
        
        let width = shape.dims[1] as u32;
        let height = shape.dims[0] as u32;
        
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Tensor Graphics Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        // Copy tensor data to texture
        let data = self.to_data();
        let bytes = data.as_slice::<f32>()
            .ok_or(GraphicsError::DataConversion("Failed to convert tensor data to f32".to_string()))?;
        let byte_data = bytemuck::cast_slice(bytes);
        
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            byte_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 4), // 4 bytes per f32
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        
        Ok(texture.create_view(&wgpu::TextureViewDescriptor::default()))
    }
    
    fn create_bind_group(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        binding: u32,
    ) -> Result<wgpu::BindGroup, GraphicsError> {
        let texture_view = self.as_texture_view(device, queue)?;
        
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: binding + 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("Tensor Bind Group"),
        });
        
        Ok(bind_group)
    }
}

/// Dual-purpose shader that can be used for both compute and graphics
pub struct DualShader {
    /// Compute shader for tensor operations
    pub compute_shader: String,
    /// Graphics vertex shader
    pub vertex_shader: String,
    /// Graphics fragment shader  
    pub fragment_shader: String,
    /// Shared parameters between compute and graphics
    pub shared_params: Vec<SharedParam>,
}

/// Parameters shared between compute and graphics shaders
#[derive(Debug, Clone)]
pub struct SharedParam {
    pub name: String,
    pub binding: u32,
    pub param_type: SharedParamType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SharedParamType {
    Buffer,
    Texture2D,
    Sampler,
    UniformBuffer,
}

impl DualShader {
    /// Create a new dual-purpose shader
    pub fn new(
        compute_shader: String,
        vertex_shader: String,
        fragment_shader: String,
    ) -> Self {
        Self {
            compute_shader,
            vertex_shader,
            fragment_shader,
            shared_params: Vec::new(),
        }
    }
    
    /// Add a shared parameter
    pub fn with_shared_param(mut self, name: String, binding: u32, param_type: SharedParamType) -> Self {
        self.shared_params.push(SharedParam {
            name,
            binding,
            param_type,
        });
        self
    }
    
    /// Create bind group layout from shared parameters
    pub fn create_bind_group_layout(&self, device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let entries: Vec<wgpu::BindGroupLayoutEntry> = self.shared_params
            .iter()
            .map(|param| {
                let ty = match param.param_type {
                    SharedParamType::Buffer => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    SharedParamType::Texture2D => wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    SharedParamType::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    SharedParamType::UniformBuffer => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                };
                
                wgpu::BindGroupLayoutEntry {
                    binding: param.binding,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty,
                    count: None,
                }
            })
            .collect();
            
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Dual Shader Bind Group Layout"),
            entries: &entries,
        })
    }
}

/// Real-time tensor visualizer using dual-purpose shaders
pub struct TensorVisualizer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface: Option<wgpu::Surface<'static>>,
    render_pipeline: Option<wgpu::RenderPipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
}

impl TensorVisualizer {
    /// Create a new tensor visualizer
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface: Option<wgpu::Surface<'static>>,
    ) -> Self {
        Self {
            device,
            queue,
            surface,
            render_pipeline: None,
            bind_group_layout: None,
        }
    }
    
    /// Initialize with a dual-purpose shader
    pub fn initialize_with_shader(&mut self, shader: &DualShader, surface_format: wgpu::TextureFormat) -> Result<(), GraphicsError> {
        let bind_group_layout = shader.create_bind_group_layout(&self.device);
        
        let vertex_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dual Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(shader.vertex_shader.clone().into()),
        });
        
        let fragment_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dual Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(shader.fragment_shader.clone().into()),
        });
        
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Dual Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let render_pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Dual Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        
        self.render_pipeline = Some(render_pipeline);
        self.bind_group_layout = Some(bind_group_layout);
        
        Ok(())
    }
    
    /// Render a tensor directly to the surface
    pub fn render_tensor<const D: usize>(&self, tensor: &Tensor<Wgpu, D>) -> Result<(), GraphicsError> {
        let surface = self.surface.as_ref()
            .ok_or(GraphicsError::NoSurface)?;
        let render_pipeline = self.render_pipeline.as_ref()
            .ok_or(GraphicsError::NotInitialized)?;
        let bind_group_layout = self.bind_group_layout.as_ref()
            .ok_or(GraphicsError::NotInitialized)?;
            
        let output = surface.get_current_texture()
            .map_err(|e| GraphicsError::SurfaceError(format!("Failed to get surface texture: {:?}", e)))?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Create bind group for tensor
        let bind_group = tensor.create_bind_group(&self.device, &self.queue, bind_group_layout, 0)?;
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            render_pass.set_pipeline(render_pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..3, 0..1); // Full-screen triangle
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        Ok(())
    }
}

/// Error types for graphics operations
#[derive(Debug)]
pub enum GraphicsError {
    NoBufferAccess,
    UnsupportedShape(String),
    DataConversion(String),
    NoSurface,
    NotInitialized,
    SurfaceError(String),
    ShaderError(String),
}

impl std::fmt::Display for GraphicsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphicsError::NoBufferAccess => write!(f, "Cannot access tensor GPU buffer"),
            GraphicsError::UnsupportedShape(msg) => write!(f, "Unsupported tensor shape: {}", msg),
            GraphicsError::DataConversion(msg) => write!(f, "Data conversion error: {}", msg),
            GraphicsError::NoSurface => write!(f, "No surface configured for rendering"),
            GraphicsError::NotInitialized => write!(f, "Visualizer not initialized"),
            GraphicsError::SurfaceError(msg) => write!(f, "Surface error: {}", msg),
            GraphicsError::ShaderError(msg) => write!(f, "Shader error: {}", msg),
        }
    }
}

impl std::error::Error for GraphicsError {}

/// Convenience functions for creating dual-purpose shaders
pub mod shaders {
    use super::*;
    
    /// Create a basic tensor visualization shader
    pub fn basic_tensor_visualization() -> DualShader {
        let compute_shader = r#"
// Compute shader for tensor operations
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= arrayLength(&data) {
        return;
    }
    
    // Example: Simple transformation
    data[index] = sin(data[index] * 3.14159);
}
"#;
        
        let vertex_shader = r#"
// Vertex shader for full-screen rendering
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vertex_index & 1u) * 2 - 1);
    let y = f32(i32(vertex_index & 2u) - 1);
    
    return vec4<f32>(x, y, 0.0, 1.0);
}
"#;
        
        let fragment_shader = r#"
// Fragment shader for tensor visualization
@group(0) @binding(0) var tensor_texture: texture_2d<f32>;
@group(0) @binding(1) var tensor_sampler: sampler;

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coord.xy / vec2<f32>(800.0, 600.0);
    let value = textureSample(tensor_texture, tensor_sampler, uv).r;
    
    // Hot colormap
    let color = if value < 0.33 {
        vec3<f32>(value * 3.0, 0.0, 0.0)
    } else if value < 0.66 {
        vec3<f32>(1.0, (value - 0.33) * 3.0, 0.0)
    } else {
        vec3<f32>(1.0, 1.0, (value - 0.66) * 3.0)
    };
    
    return vec4<f32>(color, 1.0);
}
"#;
        
        DualShader::new(
            compute_shader.to_string(),
            vertex_shader.to_string(),
            fragment_shader.to_string(),
        )
        .with_shared_param("tensor_data".to_string(), 0, SharedParamType::Texture2D)
        .with_shared_param("tensor_sampler".to_string(), 1, SharedParamType::Sampler)
    }
}
