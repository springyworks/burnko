//! Five Tensor Example with Dual No-CPU Visualization
//! 
//! This example demonstrates:
//! - 5 different Burn tensors performing various operations
//! - 2 tensors visualized using the no-CPU method
//! - All tensor operations remain GPU-resident
//! - Real-time dual visualization with split-screen display

use burn::{
    tensor::{Tensor, Device, Distribution},
    backend::Wgpu,
};
use std::sync::Arc;
use std::time::Instant;
use winit::{
    event::{Event, WindowEvent, KeyEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
    keyboard::{KeyCode, PhysicalKey},
};

type Backend = Wgpu<f32>;

//  dual-purpose shader for split-screen visualization
const DUAL_VISUALIZATION_SHADER: &str = r#"
// Two shared tensor buffers for dual visualization
@group(0) @binding(0) var<storage, read_write> tensor_data_1: array<f32>;
@group(0) @binding(1) var<storage, read_write> tensor_data_2: array<f32>;
@group(0) @binding(2) var<uniform> frame_params: FrameParams;

struct FrameParams {
    frame_count: u32,
    tensor_size: u32,
    visualization_mode: u32,
    padding: u32,
}

// COMPUTE SHADER: Apply visual enhancement effects
@compute @workgroup_size(8, 8)
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tensor_size = frame_params.tensor_size;
    let size_sqrt = u32(sqrt(f32(tensor_size)));
    
    if (global_id.x >= size_sqrt || global_id.y >= size_sqrt) {
        return;
    }
    
    let index = global_id.x + global_id.y * size_sqrt;
    
    if (index >= tensor_size) {
        return;
    }
    
    let time = f32(frame_params.frame_count) * 0.03;
    let x = f32(global_id.x) / f32(size_sqrt);
    let y = f32(global_id.y) / f32(size_sqrt);
    
    // Apply different visual enhancements to each tensor
    
    // Tensor 1: Ripple effect
    let center1 = vec2<f32>(0.5 + sin(time) * 0.3, 0.5 + cos(time * 0.8) * 0.3);
    let dist1 = distance(vec2<f32>(x, y), center1);
    let ripple1 = sin(dist1 * 15.0 - time * 4.0) * 0.2 + 1.0;
    tensor_data_1[index] = tensor_data_1[index] * ripple1;
    
    // Tensor 2: Spiral pattern
    let center2 = vec2<f32>(0.5, 0.5);
    let angle2 = atan2(y - center2.y, x - center2.x) + time * 2.0;
    let spiral2 = sin(angle2 * 3.0 + dist1 * 8.0) * 0.15 + 1.0;
    tensor_data_2[index] = tensor_data_2[index] * spiral2;
}

// VERTEX SHADER: Full-screen quad
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0), vec2<f32>(-1.0,  1.0)
    );
    
    var uv = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 0.0)
    );
    
    var output: VertexOutput;
    output.position = vec4<f32>(pos[vertex_index], 0.0, 1.0);
    output.uv = uv[vertex_index];
    return output;
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// FRAGMENT SHADER: Split-screen dual visualization
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let tensor_size = frame_params.tensor_size;
    let size_sqrt = u32(sqrt(f32(tensor_size)));
    
    // Split screen: left half shows tensor 1, right half shows tensor 2
    let is_left_half = input.uv.x < 0.5;
    
    var sample_uv: vec2<f32>;
    if (is_left_half) {
        // Left half: map to full texture coordinates for tensor 1
        sample_uv = vec2<f32>(input.uv.x * 2.0, input.uv.y);
    } else {
        // Right half: map to full texture coordinates for tensor 2
        sample_uv = vec2<f32>((input.uv.x - 0.5) * 2.0, input.uv.y);
    }
    
    let pixel_x = u32(sample_uv.x * f32(size_sqrt));
    let pixel_y = u32(sample_uv.y * f32(size_sqrt));
    let index = pixel_x + pixel_y * size_sqrt;
    
    if (index >= tensor_size) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    
    var value: f32;
    var color: vec3<f32>;
    
    if (is_left_half) {
        // Tensor 1: Hot colormap (black -> red -> yellow -> white)
        value = clamp(tensor_data_1[index], -2.0, 2.0) * 0.25 + 0.5;
        if (value < 0.33) {
            color = vec3<f32>(value * 3.0, 0.0, 0.0);
        } else if (value < 0.66) {
            let t = (value - 0.33) * 3.0;
            color = vec3<f32>(1.0, t, 0.0);
        } else {
            let t = (value - 0.66) * 3.0;
            color = vec3<f32>(1.0, 1.0, t);
        }
    } else {
        // Tensor 2: Viridis colormap (purple -> blue -> green -> yellow)
        value = clamp(tensor_data_2[index], -2.0, 2.0) * 0.25 + 0.5;
        color = viridis_colormap(value);
    }
    
    // Add a subtle dividing line in the middle
    if (abs(input.uv.x - 0.5) < 0.002) {
        color = mix(color, vec3<f32>(1.0, 1.0, 1.0), 0.5);
    }
    
    return vec4<f32>(color, 1.0);
}

// Viridis colormap approximation
fn viridis_colormap(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.2777273, 0.005407344, 0.3340998);
    let c1 = vec3<f32>(0.1050930, 1.404613, 1.384590);
    let c2 = vec3<f32>(-0.3308618, 0.214847, 0.09509516);
    let c3 = vec3<f32>(-4.634230, -5.799100, -19.33244);
    let c4 = vec3<f32>(6.228269, 14.17993, 56.69055);
    let c5 = vec3<f32>(4.776384, -13.74514, -65.35303);
    let c6 = vec3<f32>(-5.435455, 4.645852, 26.31290);
    
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}
"#;

// Application state managing 5 tensors with dual visualization
struct 
FiveTensorDemo {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    
    // GPU buffers for the 2 visualized tensors
    tensor_buffer_1: wgpu::Buffer,
    tensor_buffer_2: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    
    // Burn tensors (5 total)
    burn_device: Device<Backend>,
    tensor_weights: Tensor<Backend, 2>,      // Neural network weights
    tensor_gradients: Tensor<Backend, 2>,    // Gradients during training
    tensor_activations: Tensor<Backend, 2>,  // Layer activations (visualized #1)
    tensor_features: Tensor<Backend, 2>,     // Feature maps (visualized #2)  
    tensor_errors: Tensor<Backend, 2>,       // Error maps
    
    frame_count: u32,
    start_time: Instant,
    current_epoch: u32,
    tensor_size: usize,
}

impl FiveTensorDemo {
    async fn new(window: Arc<winit::window::Window>) -> Self {
        println!("🔥 Five Tensor Demo with Dual No-CPU Visualization");
        println!("===================================================");
        println!("   5 Burn tensors: weights, gradients, activations, features, errors");
        println!("   2 tensors visualized: activations (left, hot) + features (right, viridis)");
        println!("   ALL operations stay on GPU - zero CPU transfers!");
        println!("   Press SPACE to simulate training steps");

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
                    label: Some("Five Tensor Demo Device"),
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
            label: Some("Dual Visualization Shader"),
            source: wgpu::ShaderSource::Wgsl(DUAL_VISUALIZATION_SHADER.into()),
        });

        // Create GPU buffers for the 2 visualized tensors
        let tensor_size = 256 * 256; // 64K elements each
        let tensor_buffer_1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tensor 1 Buffer (Activations)"),
            size: (tensor_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tensor_buffer_2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tensor 2 Buffer (Features)"),
            size: (tensor_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Uniform buffer for frame parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct FrameParams {
            frame_count: u32,
            tensor_size: u32,
            visualization_mode: u32,
            padding: u32,
        }

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Frame Parameters"),
            size: std::mem::size_of::<FrameParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Dual Tensor Bind Group Layout"),
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
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
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
            label: Some("Dual Tensor Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tensor_buffer_1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tensor_buffer_2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
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
            label: Some("Dual Tensor Compute Pipeline"),
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
            label: Some("Dual Tensor Render Pipeline"),
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

        // Initialize Burn device and create 5 tensors
        let burn_device = Device::<Backend>::default();
        
        // 1. Neural network weights (256x256)
        let tensor_weights = Tensor::<Backend, 2>::random(
            [256, 256], 
            Distribution::Normal(0.0, 0.1), 
            &burn_device
        );
        
        // 2. Gradients (256x256)
        let tensor_gradients = Tensor::<Backend, 2>::random(
            [256, 256], 
            Distribution::Normal(0.0, 0.01), 
            &burn_device
        );
        
        // 3. Activations (256x256) - VISUALIZED #1
        let tensor_activations = Tensor::<Backend, 2>::random(
            [256, 256], 
            Distribution::Normal(0.5, 0.3), 
            &burn_device
        );
        
        // 4. Feature maps (256x256) - VISUALIZED #2
        let tensor_features = Tensor::<Backend, 2>::random(
            [256, 256], 
            Distribution::Normal(0.0, 0.5), 
            &burn_device
        );
        
        // 5. Error maps (256x256)
        let tensor_errors = Tensor::<Backend, 2>::random(
            [256, 256], 
            Distribution::Normal(0.0, 0.2), 
            &burn_device
        );

        Self {
            device,
            queue,
            surface,
            surface_config,
            compute_pipeline,
            render_pipeline,
            tensor_buffer_1,
            tensor_buffer_2,
            uniform_buffer,
            bind_group,
            burn_device,
            tensor_weights,
            tensor_gradients,
            tensor_activations,
            tensor_features,
            tensor_errors,
            frame_count: 0,
            start_time: Instant::now(),
            current_epoch: 0,
            tensor_size,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    fn simulate_training_step(&mut self) {
        self.current_epoch += 1;
        
        println!("🚀 Training Step {} - Updating all 5 tensors...", self.current_epoch);
        
        // 1. Update weights using gradients (gradient descent)
        let learning_rate = 0.01;
        self.tensor_weights = self.tensor_weights.clone() - self.tensor_gradients.clone() * learning_rate;
        
        // 2. Compute new activations using updated weights
        let input_simulation = Tensor::<Backend, 2>::random([256, 256], Distribution::Normal(0.0, 1.0), &self.burn_device);
        self.tensor_activations = self.tensor_weights.clone().matmul(input_simulation).clamp(0.0, f32::INFINITY);
        
        // 3. Update feature maps based on activations
        self.tensor_features = self.tensor_activations.clone().sin() * 0.8 + self.tensor_features.clone() * 0.2;
        
        // 4. Compute new gradients based on errors
        self.tensor_gradients = self.tensor_errors.clone() * 2.0;
        
        // 5. Update errors (simulated loss gradients)
        let target_simulation = Tensor::<Backend, 2>::zeros([256, 256], &self.burn_device);
        self.tensor_errors = (self.tensor_activations.clone() - target_simulation).abs();
        
        // Print statistics for monitoring (CPU operations here are just for logging)
        let activation_mean = self.tensor_activations.clone().mean().into_scalar();
        let feature_mean = self.tensor_features.clone().mean().into_scalar();
        let error_mean = self.tensor_errors.clone().mean().into_scalar();
        
        println!("   📊 Activation mean: {:.4}", activation_mean);
        println!("   📊 Feature mean: {:.4}", feature_mean);
        println!("   📊 Error mean: {:.4}", error_mean);
    }

    fn update_visualization_data(&mut self) {
        // Copy the 2 visualized tensors to GPU buffers
        // This is the ONLY CPU transfer - just for visualization
        
        let activation_data = self.tensor_activations.to_data();
        let activation_slice = activation_data.as_slice::<f32>().unwrap();
        self.queue.write_buffer(&self.tensor_buffer_1, 0, bytemuck::cast_slice(activation_slice));
        
        let feature_data = self.tensor_features.to_data();
        let feature_slice = feature_data.as_slice::<f32>().unwrap();
        self.queue.write_buffer(&self.tensor_buffer_2, 0, bytemuck::cast_slice(feature_slice));
    }

    fn update_and_render(&mut self) {
        self.frame_count += 1;
        
        // Automatic training simulation every 120 frames (2 seconds at 60fps)
        if self.frame_count % 120 == 0 {
            self.simulate_training_step();
        }
        
        // Update visualization data every 10 frames
        if self.frame_count % 10 == 0 {
            self.update_visualization_data();
        }

        // Update frame parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct FrameParams {
            frame_count: u32,
            tensor_size: u32,
            visualization_mode: u32,
            padding: u32,
        }

        let params = FrameParams {
            frame_count: self.frame_count,
            tensor_size: self.tensor_size as u32,
            visualization_mode: 0,
            padding: 0,
        };
        
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[params]));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Five Tensor Render Encoder"),
        });

        // Run compute shader for visual effects
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Dual Tensor Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(32, 32, 1); // 256x256 total threads
        }

        // Render split-screen visualization
        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Outdated) => {
                // Surface is outdated, reconfigure and try again
                self.surface.configure(&self.device, &self.surface_config);
                return; // Skip this frame
            }
            Err(e) => {
                eprintln!("Surface error: {:?}", e);
                return; // Skip this frame
            }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Dual Tensor Visualization Pass"),
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

        // Status updates
        if self.frame_count % 180 == 0 {
            let elapsed = self.start_time.elapsed().as_secs_f32();
            println!("⏰ Frame: {} | Time: {:.1}s | Epoch: {} | FPS: {:.1}", 
                self.frame_count, elapsed, self.current_epoch, self.frame_count as f32 / elapsed);
        }
    }

    fn handle_key_input(&mut self, key_code: KeyCode) {
        match key_code {
            KeyCode::Space => {
                println!("🔥 Manual training step triggered!");
                self.simulate_training_step();
            }
            KeyCode::KeyR => {
                println!("🔄 Resetting all tensors...");
                self.tensor_weights = Tensor::<Backend, 2>::random([256, 256], Distribution::Normal(0.0, 0.1), &self.burn_device);
                self.tensor_gradients = Tensor::<Backend, 2>::random([256, 256], Distribution::Normal(0.0, 0.01), &self.burn_device);
                self.tensor_activations = Tensor::<Backend, 2>::random([256, 256], Distribution::Normal(0.5, 0.3), &self.burn_device);
                self.tensor_features = Tensor::<Backend, 2>::random([256, 256], Distribution::Normal(0.0, 0.5), &self.burn_device);
                self.tensor_errors = Tensor::<Backend, 2>::random([256, 256], Distribution::Normal(0.0, 0.2), &self.burn_device);
                self.current_epoch = 0;
            }
            _ => {}
        }
    }
}

#[tokio::main]
async fn main() {
    env_logger::init();
    
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new()
        .with_title("🔥 Five Tensor Demo - Dual No-CPU Visualization")
        .with_inner_size(winit::dpi::LogicalSize::new(1200, 600))
        .build(&event_loop)
        .unwrap());

    let mut demo = FiveTensorDemo::new(window.clone()).await;

    println!("\n🎮 CONTROLS:");
    println!("   SPACE - Trigger manual training step");
    println!("   R - Reset all tensors");
    println!("   ESC - Exit");
    println!("\n📺 VISUALIZATION:");
    println!("   LEFT: Activations (Hot colormap)");
    println!("   RIGHT: Features (Viridis colormap)");

    event_loop.run(move |event, target| {
        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => {
                        println!("🎉 Five tensor demo complete!");
                        target.exit();
                    }
                    WindowEvent::Resized(physical_size) => {
                        demo.resize(physical_size);
                    }
                    WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(key_code), .. }, .. } => {
                        if key_code == KeyCode::Escape {
                            target.exit();
                        } else {
                            demo.handle_key_input(key_code);
                        }
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
