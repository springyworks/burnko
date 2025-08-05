//! Render pipeline for tensor visualization

use super::{TensorTexture, ColorMap, VideoConfig};
use std::sync::Arc;

/// Main render pipeline for tensor visualization
pub struct TensorRenderPipeline {
    /// WGPU device
    device: Arc<wgpu::Device>,
    /// Render pipeline for tensor visualization
    pipeline: wgpu::RenderPipeline,
    /// Bind group layout for textures
    bind_group_layout: wgpu::BindGroupLayout,
    /// Current colormap
    colormap: ColorMap,
}

impl TensorRenderPipeline {
    /// Create a new render pipeline
    pub fn new(
        device: Arc<wgpu::Device>,
        surface_format: wgpu::TextureFormat,
        colormap: ColorMap,
    ) -> Result<Self, PipelineError> {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tensor Visualization Bind Group Layout"),
            entries: &[
                // Tensor texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Texture sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tensor Visualization Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create shader module with colormap-specific fragment shader
        let shader_source = Self::generate_shader_source(&colormap);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tensor Visualization Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Tensor Visualization Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Ok(Self {
            device,
            pipeline,
            bind_group_layout,
            colormap,
        })
    }

    /// Render a tensor texture to the given render pass
    pub fn render(
        &self,
        tensor_texture: &TensorTexture,
        render_pass: &mut wgpu::RenderPass,
        sampler: &wgpu::Sampler,
    ) {
        // Create bind group for this tensor
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tensor Visualization Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tensor_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1); // Full-screen triangle
    }

    /// Update the pipeline with a new colormap
    pub fn update_colormap(
        &mut self,
        colormap: ColorMap,
        surface_format: wgpu::TextureFormat,
    ) -> Result<(), PipelineError> {
        if std::mem::discriminant(&self.colormap) != std::mem::discriminant(&colormap) {
            // Need to recreate pipeline with new shader
            *self = Self::new(self.device.clone(), surface_format, colormap)?;
        }
        Ok(())
    }

    /// Generate WGSL shader source with the specified colormap
    fn generate_shader_source(colormap: &ColorMap) -> String {
        let colormap_function = match colormap {
            ColorMap::Grayscale => "vec3<f32>(value, value, value)".to_string(),
            ColorMap::Viridis => include_str!("shaders/viridis.wgsl").to_string(),
            ColorMap::Plasma => include_str!("shaders/plasma.wgsl").to_string(),
            ColorMap::Hot => include_str!("shaders/hot.wgsl").to_string(),
            ColorMap::Cool => include_str!("shaders/cool.wgsl").to_string(),
            ColorMap::Jet => include_str!("shaders/jet.wgsl").to_string(),
            ColorMap::Custom { min_color, max_color } => {
                format!(
                    "mix(vec3<f32>({}, {}, {}), vec3<f32>({}, {}, {}), value)",
                    min_color[0], min_color[1], min_color[2],
                    max_color[0], max_color[1], max_color[2]
                )
            }
        };

        let colormap_call = match colormap {
            ColorMap::Grayscale | ColorMap::Custom { .. } => colormap_function,
            _ => format!("{}(value)", 
                match colormap {
                    ColorMap::Viridis => "viridis_colormap",
                    ColorMap::Plasma => "plasma_colormap", 
                    ColorMap::Hot => "hot_colormap",
                    ColorMap::Cool => "cool_colormap",
                    ColorMap::Jet => "jet_colormap",
                    _ => unreachable!()
                }
            ),
        };

        format!(r#"
// Vertex shader - generates full-screen triangle
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {{
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u),
    );
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}}

// Colormap function
{colormap_function}

// Fragment shader - samples tensor and applies colormap
@group(0) @binding(0)
var tensor_texture: texture_2d<f32>;

@group(0) @binding(1)
var tensor_sampler: sampler;

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {{
    let texture_size = textureDimensions(tensor_texture);
    let uv = position.xy / vec2<f32>(f32(texture_size.x), f32(texture_size.y));
    
    // Sample tensor value
    let tensor_value = textureSample(tensor_texture, tensor_sampler, uv).r;
    
    // Normalize value (assuming tensor values are in reasonable range)
    // TODO: Add configurable normalization parameters
    let value = clamp(tensor_value, 0.0, 1.0);
    
    // Apply colormap
    let color = {colormap_call};
    
    return vec4<f32>(color, 1.0);
}}
"#, 
            colormap_function = colormap_function,
            colormap_call = colormap_call
        )
    }
}

/// Errors that can occur in the render pipeline
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("WGPU error: {0}")]
    WgpuError(#[from] wgpu::Error),
    
    #[error("Shader compilation failed: {0}")]
    ShaderError(String),
    
    #[error("Unsupported colormap: {0}")]
    UnsupportedColormap(String),
}

/// Render context for managing rendering state
pub struct RenderContext {
    /// WGPU device
    pub device: Arc<wgpu::Device>,
    /// WGPU queue
    pub queue: Arc<wgpu::Queue>,
    /// Surface for window rendering
    pub surface: Option<wgpu::Surface>,
    /// Surface configuration
    pub surface_config: Option<wgpu::SurfaceConfiguration>,
    /// Linear sampler for tensor textures
    pub sampler: wgpu::Sampler,
}

impl RenderContext {
    /// Create a new render context
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Tensor Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            device,
            queue,
            surface: None,
            surface_config: None,
            sampler,
        }
    }

    /// Setup surface for window rendering
    pub fn setup_surface(
        &mut self,
        surface: wgpu::Surface,
        width: u32,
        height: u32,
    ) -> Result<(), PipelineError> {
        let surface_caps = surface.get_capabilities(&self.device.get_info().name);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&self.device, &config);
        
        self.surface = Some(surface);
        self.surface_config = Some(config);
        
        Ok(())
    }

    /// Get the surface format
    pub fn surface_format(&self) -> Option<wgpu::TextureFormat> {
        self.surface_config.as_ref().map(|config| config.format)
    }
}
