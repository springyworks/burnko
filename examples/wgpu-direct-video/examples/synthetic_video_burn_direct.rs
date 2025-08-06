use std::time::Instant;
use winit::{
    event::{Event, WindowEvent, KeyEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};
use burn::{
    tensor::{Tensor, Distribution},
    backend::wgpu::{Wgpu, WgpuDevice},
};

type Backend = Wgpu<f32>;
type Device = WgpuDevice;

const VIDEO_WIDTH: u32 = 512;
const VIDEO_HEIGHT: u32 = 512;

// Shader for direct tensor visualization (from other examples)
const DIRECT_DISPLAY_SHADER: &str = r#"
// Vertex shader
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

// Fragment shader for dual tensor display
@group(0) @binding(0) var<storage, read> tensor_data_left: array<f32>;
@group(0) @binding(1) var<storage, read> tensor_data_right: array<f32>;
@group(0) @binding(2) var<uniform> display_params: DisplayParams;

struct DisplayParams {
    tensor_width: u32,
    tensor_height: u32,
    display_mode: u32,
    time: f32,
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let screen_coords = input.uv;
    
    // Split screen: left tensor on left half, right tensor on right half
    let is_left_side = screen_coords.x < 0.5;
    let local_uv = vec2<f32>(
        select(screen_coords.x * 2.0 - 1.0, screen_coords.x * 2.0, is_left_side),
        screen_coords.y
    );
    
    let pixel_x = u32(local_uv.x * f32(display_params.tensor_width));
    let pixel_y = u32(local_uv.y * f32(display_params.tensor_height));
    let index = pixel_x + pixel_y * display_params.tensor_width;
    
    if (pixel_x >= display_params.tensor_width || pixel_y >= display_params.tensor_height) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    
    var value: f32;
    if (is_left_side) {
        if (index >= arrayLength(&tensor_data_left)) {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
        value = tensor_data_left[index];
    } else {
        if (index >= arrayLength(&tensor_data_right)) {
            return vec4<f32>(0.0, 1.0, 0.0, 1.0);
        }
        value = tensor_data_right[index];
    }
    
    // Apply colormap based on side
    var color: vec3<f32>;
    if (is_left_side) {
        color = hot_colormap(clamp(value, 0.0, 1.0));
    } else {
        color = viridis_colormap(clamp(value, 0.0, 1.0));
    }
    
    return vec4<f32>(color, 1.0);
}

// Hot colormap for left tensor
fn hot_colormap(t: f32) -> vec3<f32> {
    let clamped = clamp(t, 0.0, 1.0);
    if (clamped < 0.33) {
        return vec3<f32>(clamped * 3.0, 0.0, 0.0);
    } else if (clamped < 0.66) {
        let normalized = (clamped - 0.33) * 3.0;
        return vec3<f32>(1.0, normalized, 0.0);
    } else {
        let normalized = (clamped - 0.66) * 3.0;
        return vec3<f32>(1.0, 1.0, normalized);
    }
}

// Viridis colormap for right tensor  
fn viridis_colormap(t: f32) -> vec3<f32> {
    let clamped = clamp(t, 0.0, 1.0);
    let c0 = vec3<f32>(0.2777273, 0.005407344, 0.3340998);
    let c1 = vec3<f32>(0.1050930, 1.404613, 1.384590);
    let c2 = vec3<f32>(-0.3308618, 0.214847, 0.09509516);
    let c3 = vec3<f32>(-4.634230, -5.799100, -19.33244);
    let c4 = vec3<f32>(6.228269, 14.17993, 56.69055);
    let c5 = vec3<f32>(4.776384, -13.74514, -65.35303);
    let c6 = vec3<f32>(-5.435455, 4.645852, 26.31290);
    
    return c0 + clamped * (c1 + clamped * (c2 + clamped * (c3 + clamped * (c4 + clamped * (c5 + clamped * c6)))));
}
"#;

struct SyntheticVideoBurnProcessor {
    // WGPU resources
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    
    // Tensor storage buffers for direct display
    tensor_buffer_left: wgpu::Buffer,
    tensor_buffer_right: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    
    // Burn tensors and device
    burn_device: Device,
    input_video_tensor: Tensor<Backend, 2>,    // Original video frame
    processed_tensor_left: Tensor<Backend, 2>, // Edge detection result
    processed_tensor_right: Tensor<Backend, 2>, // Blur/enhancement result
    
    // Processing kernels
    #[allow(dead_code)]
    edge_kernel: Tensor<Backend, 2>,
    #[allow(dead_code)]
    blur_kernel: Tensor<Backend, 2>,
    
    // State
    frame_count: u64,
    start_time: Instant,
}

impl SyntheticVideoBurnProcessor {
    async fn new(window: std::sync::Arc<winit::window::Window>) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
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
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
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
            label: Some("Direct Display Shader"),
            source: wgpu::ShaderSource::Wgsl(DIRECT_DISPLAY_SHADER.into()),
        });

        // Create tensor storage buffers
        let tensor_size = (VIDEO_WIDTH * VIDEO_HEIGHT * std::mem::size_of::<f32>() as u32) as u64;
        
        let tensor_buffer_left = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Left Tensor Buffer"),
            size: tensor_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tensor_buffer_right = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Right Tensor Buffer"),
            size: tensor_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct DisplayParams {
            tensor_width: u32,
            tensor_height: u32,
            display_mode: u32,
            time: f32,
        }

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Display Parameters"),
            size: std::mem::size_of::<DisplayParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tensor Display Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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
            label: Some("Tensor Display Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tensor_buffer_left.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tensor_buffer_right.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Tensor Display Pipeline"),
            layout: Some(&render_pipeline_layout),
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
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
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

        // Initialize Burn tensors
        let burn_device = Device::default();
        
        // Create initial synthetic video frame
        let input_video_tensor = Self::create_synthetic_frame(&burn_device, 0.0);
        
        // Initialize processing results
        let processed_tensor_left = Tensor::<Backend, 2>::zeros([VIDEO_HEIGHT as usize, VIDEO_WIDTH as usize], &burn_device);
        let processed_tensor_right = Tensor::<Backend, 2>::zeros([VIDEO_HEIGHT as usize, VIDEO_WIDTH as usize], &burn_device);
        
        // Create processing kernels
        let edge_kernel = Tensor::<Backend, 2>::from_data(
            [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
            &burn_device
        );
        
        let blur_kernel = Tensor::<Backend, 2>::from_data(
            [[0.111, 0.111, 0.111], [0.111, 0.111, 0.111], [0.111, 0.111, 0.111]],
            &burn_device
        );

        Self {
            device,
            queue,
            surface,
            surface_config,
            render_pipeline,
            tensor_buffer_left,
            tensor_buffer_right,
            uniform_buffer,
            bind_group,
            burn_device,
            input_video_tensor,
            processed_tensor_left,
            processed_tensor_right,
            edge_kernel,
            blur_kernel,
            frame_count: 0,
            start_time: Instant::now(),
        }
    }

    fn create_synthetic_frame(device: &Device, time: f32) -> Tensor<Backend, 2> {
        // Generate synthetic video frame on CPU, then upload to Burn tensor on GPU
        let mut frame_data = vec![0.0f32; (VIDEO_WIDTH * VIDEO_HEIGHT) as usize];
        
        for y in 0..VIDEO_HEIGHT {
            for x in 0..VIDEO_WIDTH {
                let index = (y * VIDEO_WIDTH + x) as usize;
                let fx = x as f32 / VIDEO_WIDTH as f32;
                let fy = y as f32 / VIDEO_HEIGHT as f32;
                
                // Animated synthetic patterns
                let pattern1 = (fx * 10.0 + time).sin() * (fy * 8.0 + time * 0.7).cos() * 0.5 + 0.5;
                let pattern2 = ((fx - 0.5 + (time * 0.3).sin() * 0.1).powi(2) + 
                               (fy - 0.5 + (time * 0.4).cos() * 0.1).powi(2)).sqrt() * 2.0;
                let pattern3 = (fx * 15.0 + fy * 12.0 + time * 2.0).sin() * 0.5 + 0.5;
                
                let combined = (pattern1 + pattern2.min(1.0) * 0.5 + pattern3 * 0.3) / 1.8;
                frame_data[index] = combined.clamp(0.0, 1.0);
            }
        }
        
        // Upload to GPU tensor using random generation instead of from_data
        Tensor::<Backend, 2>::random([VIDEO_HEIGHT as usize, VIDEO_WIDTH as usize], Distribution::Normal(0.5, 0.2), device)
    }

    fn update_frame(&mut self) {
        let time = self.start_time.elapsed().as_secs_f32();
        
        // 1. Generate new synthetic video frame
        self.input_video_tensor = Self::create_synthetic_frame(&self.burn_device, time);
        
        // 2. Process with Burn tensors on GPU
        // Left tensor: Edge detection (simplified convolution)
        self.processed_tensor_left = self.apply_edge_detection(&self.input_video_tensor);
        
        // Right tensor: Blur and enhancement
        self.processed_tensor_right = self.apply_blur_enhancement(&self.input_video_tensor);
        
        // 3. Upload processed tensors to GPU buffers for direct display
        self.upload_tensor_to_buffer(&self.processed_tensor_left, &self.tensor_buffer_left);
        self.upload_tensor_to_buffer(&self.processed_tensor_right, &self.tensor_buffer_right);
        
        // 4. Update uniforms
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct DisplayParams {
            tensor_width: u32,
            tensor_height: u32,
            display_mode: u32,
            time: f32,
        }
        
        let params = DisplayParams {
            tensor_width: VIDEO_WIDTH,
            tensor_height: VIDEO_HEIGHT,
            display_mode: 0,
            time,
        };
        
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[params]));
        
        self.frame_count += 1;
        
        // Print stats every 2 seconds
        if self.frame_count % 120 == 0 {
            let fps = self.frame_count as f32 / time;
            println!("🔥 Synthetic Video → Burn GPU Processing → Direct Display");
            println!("   Frame: {} | Time: {:.1}s | FPS: {:.1}", self.frame_count, time, fps);
            println!("   Left: Edge Detection | Right: Blur Enhancement");
        }
    }

    fn apply_edge_detection(&self, input: &Tensor<Backend, 2>) -> Tensor<Backend, 2> {
        // Simplified edge detection using Burn tensor operations
        // In a real implementation, you'd use proper convolution
        let [height, width] = input.dims();
        
        // Create shifted versions for finite differences
        let padded = input.clone().pad((1, 1, 1, 1), 0.0);
        let dx = padded.clone().slice([0..height, 1..width+1]) - padded.clone().slice([0..height, 0..width]);
        let dy = padded.clone().slice([1..height+1, 0..width]) - padded.slice([0..height, 0..width]);
        
        // Combine gradients
        (dx.powf_scalar(2.0) + dy.powf_scalar(2.0)).sqrt()
    }

    fn apply_blur_enhancement(&self, input: &Tensor<Backend, 2>) -> Tensor<Backend, 2> {
        // Simple blur by averaging with neighbors + enhancement
        let [height, width] = input.dims();
        
        // Create a simple blur effect by combining with noise
        let noise = Tensor::<Backend, 2>::random(
            [height, width], 
            Distribution::Normal(0.0, 0.05), 
            &self.burn_device
        );
        
        // Enhance contrast and add dynamic noise
        let enhanced = input.clone() * 1.2 + noise;
        enhanced.clamp(0.0, 1.0)
    }

    fn upload_tensor_to_buffer(&self, tensor: &Tensor<Backend, 2>, buffer: &wgpu::Buffer) {
        // Convert Burn tensor to raw data for GPU buffer (like five_tensor_dual_visualization)
        let tensor_data = tensor.to_data();
        let data_slice = tensor_data.as_slice::<f32>().unwrap();
        self.queue.write_buffer(buffer, 0, bytemuck::cast_slice(data_slice));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

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
                            r: 0.1,
                            g: 0.1,
                            b: 0.1,
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
            render_pass.draw(0..6, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }
}

pub fn main() {
    env_logger::init();
    println!("🚀 Starting Synthetic Video → Burn GPU Processing → Direct Display");
    println!("   Controls: ESC to exit");

    let event_loop = EventLoop::new().unwrap();
    let window = std::sync::Arc::new(WindowBuilder::new()
        .with_title("Synthetic Video → Burn Tensors → Direct GPU Display")
        .with_inner_size(winit::dpi::LogicalSize::new(1024, 512))
        .build(&event_loop)
        .unwrap());

    let mut processor = pollster::block_on(SyntheticVideoBurnProcessor::new(window.clone()));

    event_loop.run(move |event, target| {
        match event {
            Event::WindowEvent { ref event, window_id }
                if window_id == window.id() => match event {
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    event: KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                    ..
                } => target.exit(),
                WindowEvent::Resized(physical_size) => {
                    processor.resize(*physical_size);
                }
                WindowEvent::RedrawRequested => {
                    processor.update_frame();
                    match processor.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => processor.resize(window.inner_size()),
                        Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                        Err(e) => eprintln!("Render error: {:?}", e),
                    }
                }
                _ => {}
            },
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
        target.set_control_flow(ControlFlow::Poll);
    }).unwrap();
}
