//! Single Shader Dual-Purpose Demo
//! 
//! This demonstrates the revolutionary concept: ONE SHADER used for BOTH:
//! 1. Tensor computation (compute shader)
//! 2. Real-time visualization (fragment shader)
//! 
//! The same GPU memory, same shader logic, dual purpose!

use burn_wgpu::WgpuDevice;
use std::sync::Arc;
use std::time::Instant;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

type Backend = burn_wgpu::Wgpu<f32>;

// THE MAGIC: Single WGSL shader source used for BOTH compute AND graphics!
const DUAL_PURPOSE_SHADER: &str = r#"
// ============================================================================
// DUAL-PURPOSE SHADER: Used for BOTH tensor computation AND visualization!
// ============================================================================

// Shared data structure - same memory used by compute and graphics
@group(0) @binding(0) var<storage, read_write> tensor_data: array<f32>;
@group(0) @binding(1) var<uniform> frame_count: u32;

// ============================================================================
// COMPUTE SHADER SECTION: Tensor operations
// ============================================================================
@compute @workgroup_size(8, 8)
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x + global_id.y * 1024u;
    
    if (index >= arrayLength(&tensor_data)) {
        return;
    }
    
    // DRAMATIC VISUAL EFFECTS - Much more noticeable!
    let time = f32(frame_count) * 0.05;
    let x = f32(global_id.x) / 1024.0;
    let y = f32(global_id.y) / 1024.0;
    
    // Create moving ripples and spirals
    let center_x = 0.5 + sin(time) * 0.3;
    let center_y = 0.5 + cos(time * 0.7) * 0.3;
    let dist = distance(vec2<f32>(x, y), vec2<f32>(center_x, center_y));
    
    // Animated ripple effect
    let ripple = sin(dist * 20.0 - time * 5.0) * 0.5 + 0.5;
    
    // Rotating spiral pattern
    let angle = atan2(y - center_y, x - center_x) + time;
    let spiral = sin(angle * 4.0 + dist * 10.0) * 0.5 + 0.5;
    
    // Combine effects for maximum visual impact
    let final_value = mix(ripple, spiral, sin(time * 2.0) * 0.5 + 0.5);
    
    tensor_data[index] = final_value;
}

// ============================================================================
// GRAPHICS SHADER SECTION: Visualization
// ============================================================================

// Vertex shader for full-screen quad
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    // Generate full-screen quad vertices
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0)
    );
    return vec4<f32>(pos[vertex_index], 0.0, 1.0);
}

// Fragment shader for hot colormap visualization
@fragment
fn fs_main(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
    let index = u32(coord.x) + u32(coord.y) * 1024u;
    
    if (index >= arrayLength(&tensor_data)) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    
    let value = tensor_data[index];
    
    // Hot colormap: black -> red -> yellow -> white
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

// ============================================================================
// Application State
// ============================================================================

struct SingleShaderDemo {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    
    // The magic: SAME shader module used for both pipelines!
    shader_module: wgpu::ShaderModule,
    
    // Dual pipelines using the SAME shader
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    
    // Shared resources
    tensor_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    
    // Animation state
    burn_device: WgpuDevice,
    frame_count: u32,
    start_time: Instant,
}

impl SingleShaderDemo {
    async fn new(window: Arc<winit::window::Window>) -> Self {
        println!("🔥 REVOLUTIONARY CONCEPT: Single Shader Dual-Purpose!");
        println!("   This is what you asked for - ONE shader that does BOTH:");
        println!("   1. Tensor operations (compute shader)");
        println!("   2. Live visualization (graphics shader)");
        println!("   Same GPU memory, same shader file, dual purpose!");

        // Initialize WGPU
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
                    label: Some("Dual-Purpose Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        // Surface configuration
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

        // Create THE shader module - used for BOTH compute and graphics!
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dual-Purpose Tensor Shader"),
            source: wgpu::ShaderSource::Wgsl(DUAL_PURPOSE_SHADER.into()),
        });

        // Create tensor data buffer (shared between compute and graphics)
        let tensor_size = 1024 * 1024; // 1M elements
        let tensor_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shared Tensor Buffer"),
            size: (tensor_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create uniform buffer for frame counter
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Frame Counter Uniform"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Dual-Purpose Bind Group Layout"),
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

        // Create bind group (shared between both pipelines!)
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dual-Purpose Bind Group"),
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

        // Create compute pipeline
        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dual-Purpose Compute Pipeline"),
            layout: Some(&compute_layout),
            module: &shader_module, // SAME module!
            entry_point: "compute_main",
        });

        // Create render pipeline
        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Dual-Purpose Render Pipeline"),
            layout: Some(&render_layout),
            vertex: wgpu::VertexState {
                module: &shader_module, // SAME module!
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module, // SAME module!
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

        // Initialize Burn device for compatibility
        let burn_device = WgpuDevice::default();

        Self {
            device,
            queue,
            surface,
            surface_config,
            shader_module,
            compute_pipeline,
            render_pipeline,
            tensor_buffer,
            uniform_buffer,
            bind_group,
            burn_device,
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

    fn update_and_render(&mut self) {
        self.frame_count += 1;
        let current_time = self.start_time.elapsed().as_secs_f32();
        
        // Print dramatic update info every 60 frames
        if self.frame_count % 60 == 0 {
            println!("🌊 DRAMATIC WAVE UPDATE! Frame: {} | Time: {:.2}s | Creating ripples and spirals...", 
                self.frame_count, current_time);
        }
        
        // STEP 1: Run COMPUTE shader to update tensor data
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Dual-Purpose Command Encoder"),
        });

        // Update uniform buffer with current frame count
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.frame_count]));

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tensor Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(128, 128, 1); // 1024x1024 total threads
        }

        // STEP 2: Run GRAPHICS shader to visualize the same data
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
            render_pass.draw(0..6, 0..1); // Full-screen quad
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

// ============================================================================
// Main Application Loop
// ============================================================================

#[tokio::main]
async fn main() {
    env_logger::init();
    
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new()
        .with_title("🔥 Single Shader Dual-Purpose Demo - Revolutionary!")
        .with_inner_size(winit::dpi::LogicalSize::new(1024, 1024))
        .build(&event_loop)
        .unwrap());

    let mut demo = SingleShaderDemo::new(window.clone()).await;

    event_loop.run(move |event, target| {
        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => {
                        println!("🎉 Revolutionary single shader demo complete!");
                        target.exit();
                    }
                    WindowEvent::Resized(physical_size) => {
                        demo.resize(physical_size);
                    }
                    WindowEvent::RedrawRequested => {
                        demo.update_and_render();
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
    }).unwrap();
}
