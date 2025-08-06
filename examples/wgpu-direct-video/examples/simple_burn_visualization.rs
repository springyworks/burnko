//! Simple Burn Tensor Visualization using Single Dual-Purpose Shader
//! 
//! This demonstrates regular Burn tensor operations with the
//! single shader dual-purpose visualization concept.

use burn::{
    tensor::{Tensor, Device},
    backend::Wgpu,
};
use std::sync::Arc;
use std::time::Instant;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

type Backend = Wgpu<f32>;

// Same shader from our successful demo!
const DUAL_PURPOSE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> tensor_data: array<f32>;
@group(0) @binding(1) var<uniform> frame_count: u32;

// COMPUTE SHADER: For tensor operations
@compute @workgroup_size(8, 8)
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x + global_id.y * 512u;
    
    if (index >= arrayLength(&tensor_data)) {
        return;
    }
    
    // Burn tensor operations create the data, 
    // we just apply simple visual effects for display
    let time = f32(frame_count) * 0.02;
    let x = f32(global_id.x) / 512.0;
    let y = f32(global_id.y) / 512.0;
    
    let center_x = 0.5 + sin(time) * 0.2;
    let center_y = 0.5 + cos(time * 0.7) * 0.2;
    let dist = distance(vec2<f32>(x, y), vec2<f32>(center_x, center_y));
    
    let enhancement = sin(dist * 15.0 - time * 3.0) * 0.1 + 1.0;
    tensor_data[index] = tensor_data[index] * enhancement;
}

// VERTEX SHADER: Full-screen quad
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0), vec2<f32>(-1.0,  1.0)
    );
    return vec4<f32>(pos[vertex_index], 0.0, 1.0);
}

// FRAGMENT SHADER: Hot colormap visualization
@fragment
fn fs_main(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
    let index = u32(coord.x) + u32(coord.y) * 512u;
    
    if (index >= arrayLength(&tensor_data)) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    
    let value = clamp(tensor_data[index], -1.0, 1.0) * 0.5 + 0.5;
    
    var color: vec3<f32>;
    if (value < 0.33) {
        color = vec3<f32>(value * 3.0, 0.0, 0.0);
    } else if (value < 0.66) {
        let t = (value - 0.33) * 3.0;
        color = vec3<f32>(1.0, t, 0.0);
    } else {
        let t = (value - 0.66) * 3.0;
        color = vec3<f32>(1.0, 1.0, t);
    }
    
    return vec4<f32>(color, 1.0);
}
"#;

struct SimpleTensorViz {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    
    tensor_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    
    // Burn tensors
    burn_device: Device<Backend>,
    current_tensor: Tensor<Backend, 2>,
    frame_count: u32,
    start_time: Instant,
}

impl SimpleTensorViz {
    async fn new(window: Arc<winit::window::Window>) -> Self {
        println!("🔥 Simple Burn Tensor + Dual Shader Visualization");
        println!("   Regular Burn operations with single shader visualization!");

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();
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
                    label: Some("Simple Tensor Viz Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_capabilities(&adapter).formats[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dual-Purpose Tensor Shader"),
            source: wgpu::ShaderSource::Wgsl(DUAL_PURPOSE_SHADER.into()),
        });

        let tensor_size = 512 * 512;
        let tensor_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tensor Data Buffer"),
            size: (tensor_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Frame Counter Uniform"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tensor Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tensor Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tensor_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tensor Compute Pipeline"),
            layout: Some(&compute_layout),
            module: &shader_module,
            entry_point: "compute_main",
        });

        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Tensor Render Pipeline"),
            layout: Some(&render_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
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

        // Create Burn device and initial tensor
        let burn_device = Device::<Backend>::default();
        let current_tensor = Tensor::<Backend, 2>::random([512, 512], burn::tensor::Distribution::Normal(0.0, 1.0), &burn_device);

        Self {
            device,
            queue,
            surface,
            surface_config,
            compute_pipeline,
            render_pipeline,
            tensor_buffer,
            uniform_buffer,
            bind_group,
            burn_device,
            current_tensor,
            frame_count: 0,
            start_time: Instant::now(),
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    fn update_burn_tensor(&mut self) {
        // Regular Burn tensor operations!
        match self.frame_count % 3 {
            0 => {
                // Matrix operations
                let transform = Tensor::<Backend, 2>::random([512, 512], burn::tensor::Distribution::Normal(0.0, 0.1), &self.burn_device);
                self.current_tensor = self.current_tensor.clone().matmul(transform);
                println!("🔥 Burn: Applied matrix multiplication");
            },
            1 => {
                // Element-wise trigonometric functions
                self.current_tensor = self.current_tensor.clone().sin();
                println!("🔥 Burn: Applied sin() operation");
            },
            2 => {
                // Normalization
                let mean = self.current_tensor.clone().mean().into_scalar();
                let std = self.current_tensor.clone().var(1).sqrt().mean().into_scalar();
                self.current_tensor = (self.current_tensor.clone() - mean) / (std + 1e-8);
                println!("🔥 Burn: Applied normalization");
            },
            _ => unreachable!(),
        }

        // Clamp for visualization
        self.current_tensor = self.current_tensor.clone().clamp(-2.0, 2.0);
    }

    fn update_and_render(&mut self) {
        self.frame_count += 1;
        
        // Do regular Burn tensor operations
        if self.frame_count % 20 == 0 {
            self.update_burn_tensor();
        }

        // Copy Burn tensor data to GPU buffer
        let tensor_data = self.current_tensor.to_data();
        let tensor_bytes = tensor_data.as_slice::<f32>().unwrap();
        let byte_data = bytemuck::cast_slice(tensor_bytes);
        self.queue.write_buffer(&self.tensor_buffer, 0, byte_data);

        // Update frame counter
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.frame_count]));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Run compute shader for visual effects
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tensor Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(64, 64, 1);
        }

        // Render visualization
        let output = self.surface.get_current_texture().unwrap();
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Tensor Visualization Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // Print stats periodically
        if self.frame_count % 60 == 0 {
            let elapsed = self.start_time.elapsed().as_secs_f32();
            let min_val = self.current_tensor.clone().min().into_scalar();
            let max_val = self.current_tensor.clone().max().into_scalar();
            let mean_val = self.current_tensor.clone().mean().into_scalar();
            println!("📊 Frame: {} | Time: {:.1}s | Tensor stats - Min: {:.3}, Max: {:.3}, Mean: {:.3}", 
                self.frame_count, elapsed, min_val, max_val, mean_val);
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let event_loop = EventLoop::new()?;
    let window = Arc::new(WindowBuilder::new()
        .with_title("🔥 Simple Burn + Dual Shader Visualization")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
        .build(&event_loop)?);

    let mut app = SimpleTensorViz::new(window.clone()).await;

    event_loop.run(move |event, target| {
        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => {
                        println!("🎉 Simple tensor visualization complete!");
                        target.exit();
                    }
                    WindowEvent::Resized(physical_size) => {
                        app.resize(physical_size);
                    }
                    WindowEvent::RedrawRequested => {
                        app.update_and_render();
                        window.request_redraw();
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}
