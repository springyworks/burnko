//! Actual tensor-to-window rendering example
//! 
//! This example demonstrates real tensor visualization in a window with actual rendering

use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    tensor::Tensor,
};
use std::time::Instant;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};
use wgpu::util::DeviceExt;

type Backend = Wgpu<f32>;

fn main() {
    println!("🎨 WGPU Direct Video Pipeline - Real Rendering Example");
    
    pollster::block_on(run());
}

async fn run() {
    // Create event loop and window
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Burn Tensor Visualization - Real Rendering")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(&window).await;
    println!("✅ WGPU renderer initialized with actual rendering pipeline");

    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => {
                    println!("🏁 Window closed - Final stats: {} frames rendered", state.frame_count);
                    elwt.exit();
                }
                WindowEvent::Resized(physical_size) => {
                    println!("📐 Window resized to: {}x{}", physical_size.width, physical_size.height);
                    state.resize(*physical_size);
                }
                WindowEvent::RedrawRequested => {
                    state.update();
                    match state.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                        Err(e) => eprintln!("⚠️ Render error: {:?}", e),
                    }
                }
                _ => {}
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    }).unwrap();
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    
    // Burn-specific
    burn_device: WgpuDevice,
    
    // Animation state
    start_time: Instant,
    frame_count: u64,
    colormap_index: usize,
    
    // GPU resources for tensor visualization
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

impl State {
    async fn new(window: &Window) -> State {
        let size = window.inner_size();

        // Create WGPU instance and surface
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create texture for tensor data
        let texture_size = 256u32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("Tensor Texture"),
            view_formats: &[],
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create bind group layout and bind group
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("Texture Bind Group Layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("Tensor Bind Group"),
        });

        // Create shader and render pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tensor Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tensor_render.wgsl").into()),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
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

        // Initialize Burn device
        let burn_device = WgpuDevice::default();

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            burn_device,
            start_time: Instant::now(),
            frame_count: 0,
            colormap_index: 0,
            texture,
            texture_view,
            bind_group,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn update(&mut self) {
        self.frame_count += 1;
        
        // Update colormap every 180 frames
        if self.frame_count % 180 == 0 {
            self.colormap_index = (self.colormap_index + 1) % 5;
            let colormap = match self.colormap_index {
                0 => "Viridis",
                1 => "Plasma",
                2 => "Hot",
                3 => "Cool",
                _ => "Jet",
            };
            println!("🎨 Switched to colormap: {}", colormap);
        }
        
        // Print FPS every 60 frames
        if self.frame_count % 60 == 0 {
            let elapsed = self.start_time.elapsed().as_secs_f32();
            let fps = self.frame_count as f32 / elapsed;
            println!("📊 Frame {}: {:.1} FPS", self.frame_count, fps);
        }

        // Generate new tensor data
        let t = self.start_time.elapsed().as_secs_f32();
        let tensor = create_dynamic_pattern(t, &self.burn_device);
        
        // Print tensor stats occasionally
        if self.frame_count % 120 == 0 {
            let tensor_stats = analyze_tensor(&tensor);
            println!("🔢 Tensor stats: {}", tensor_stats);
        }

        // Convert tensor to GPU texture data
        self.update_texture_from_tensor(&tensor);
    }
    
    fn update_texture_from_tensor(&mut self, tensor: &Tensor<Backend, 2>) {
        // Convert tensor to f32 data
        let data = tensor.clone().into_data().convert::<f32>().value;
        
        // Convert to bytes for GPU upload
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        
        // Upload to GPU texture
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4), // 256 pixels * 4 bytes per f32
                rows_per_image: Some(256),
            },
            wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..3, 0..1); // Full-screen triangle
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

/// Create a dynamic 2D pattern that evolves over time
fn create_dynamic_pattern(t: f32, device: &WgpuDevice) -> Tensor<Backend, 2> {
    let size = 256;
    let mut data = Vec::with_capacity(size * size);
    
    for y in 0..size {
        for x in 0..size {
            let x_norm = x as f32 / size as f32 * 8.0;
            let y_norm = y as f32 / size as f32 * 8.0;
            
            // Complex interference pattern
            let wave1 = (x_norm + t * 2.0).sin() * 0.3;
            let wave2 = (y_norm + t * 1.5).sin() * 0.3;
            let wave3 = ((x_norm + y_norm) * 0.7 + t * 1.2).sin() * 0.2;
            let ripple = (((x_norm - 4.0).powi(2) + (y_norm - 4.0).powi(2)).sqrt() - t * 2.0).sin() * 0.2;
            
            let value = (wave1 + wave2 + wave3 + ripple) * 0.5 + 0.5;
            data.push(value.clamp(0.0, 1.0));
        }
    }
    
    Tensor::<Backend, 1>::from_floats(data.as_slice(), device).reshape([size, size])
}

/// Analyze tensor for statistics
fn analyze_tensor(tensor: &Tensor<Backend, 2>) -> String {
    let mean = tensor.clone().mean().into_scalar();
    let min_val = tensor.clone().min().into_scalar();
    let max_val = tensor.clone().max().into_scalar();
    
    format!("mean: {:.3}, range: [{:.3}, {:.3}]", mean, min_val, max_val)
}
