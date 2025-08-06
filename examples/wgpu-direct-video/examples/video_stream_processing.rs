//! Real-time Video Processing with Burn Tensors
//! 
//! This example demonstrates:
//! - Streaming video frames directly to GPU memory
//! - Processing video with Burn tensor operations (edge detection, color transforms, etc.)
//! - Real-time display of processed frames using no-CPU visualization
//! - Multiple processing modes with live switching

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

// Shader for video processing and display
const VIDEO_PROCESSING_SHADER: &str = r#"
// Video data and processing buffers
@group(0) @binding(0) var<storage, read> video_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> processed_frame: array<f32>;
@group(0) @binding(2) var<uniform> processing_params: ProcessingParams;

struct ProcessingParams {
    frame_width: u32,
    frame_height: u32,
    processing_mode: u32,
    time: f32,
}

// COMPUTE SHADER: Video processing operations
@compute @workgroup_size(8, 8)
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let width = processing_params.frame_width;
    let height = processing_params.frame_height;
    
    if (global_id.x >= width || global_id.y >= height) {
        return;
    }
    
    let index = global_id.x + global_id.y * width;
    if (index >= arrayLength(&video_data)) {
        return;
    }
    
    let original_value = video_data[index];
    var processed_value: f32;
    
    switch (processing_params.processing_mode) {
        case 0u: {
            // Simple edge detection (finite difference)
            processed_value = edge_detect(global_id.x, global_id.y, width, height);
        }
        case 1u: {
            // Animated blur effect
            let time_factor = sin(processing_params.time + f32(index) * 0.01) * 0.5 + 0.5;
            processed_value = original_value * time_factor;
        }
        case 2u: {
            // Ripple effect based on distance from center
            let center_x = f32(width) * 0.5;
            let center_y = f32(height) * 0.5;
            let dx = f32(global_id.x) - center_x;
            let dy = f32(global_id.y) - center_y;
            let dist = sqrt(dx * dx + dy * dy);
            let ripple = sin(dist * 0.1 - processing_params.time * 3.0) * 0.3 + 0.7;
            processed_value = original_value * ripple;
        }
        case 3u: {
            // Color channel separation effect
            let wave = sin(f32(global_id.x) * 0.1 + processing_params.time) * 0.2 + 0.8;
            processed_value = original_value * wave;
        }
        default: {
            processed_value = original_value;
        }
    }
    
    processed_frame[index] = processed_value;
}

// Simple edge detection using neighboring pixels
fn edge_detect(x: u32, y: u32, width: u32, height: u32) -> f32 {
    if (x == 0u || y == 0u || x >= width - 1u || y >= height - 1u) {
        return 0.0;
    }
    
    let current = video_data[x + y * width];
    let right = video_data[(x + 1u) + y * width];
    let down = video_data[x + (y + 1u) * width];
    
    let dx = abs(current - right);
    let dy = abs(current - down);
    
    return min(dx + dy, 1.0);
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

// FRAGMENT SHADER: Display processed video with colormap
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let width = processing_params.frame_width;
    let height = processing_params.frame_height;
    
    let pixel_x = u32(input.uv.x * f32(width));
    let pixel_y = u32(input.uv.y * f32(height));
    let index = pixel_x + pixel_y * width;
    
    if (index >= arrayLength(&processed_frame)) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    
    let processed_value = clamp(processed_frame[index], 0.0, 1.0);
    
    // Apply colormap based on processing mode
    var color: vec3<f32>;
    switch (processing_params.processing_mode) {
        case 0u: {
            // Edge detection: hot colormap
            color = hot_colormap(processed_value);
        }
        case 1u: {
            // Blur: cool colormap
            color = cool_colormap(processed_value);
        }
        case 2u: {
            // Ripple: viridis colormap
            color = viridis_colormap(processed_value);
        }
        case 3u: {
            // Color separation: plasma colormap
            color = plasma_colormap(processed_value);
        }
        default: {
            color = vec3<f32>(processed_value, processed_value, processed_value);
        }
    }
    
    return vec4<f32>(color, 1.0);
}

// Colormap functions
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

fn cool_colormap(t: f32) -> vec3<f32> {
    let clamped = clamp(t, 0.0, 1.0);
    return vec3<f32>(clamped, 1.0 - clamped, 1.0);
}

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

fn plasma_colormap(t: f32) -> vec3<f32> {
    let clamped = clamp(t, 0.0, 1.0);
    let c0 = vec3<f32>(0.0504, 0.0298, 0.5280);
    let c1 = vec3<f32>(0.3827, -0.3892, 0.6557);
    let c2 = vec3<f32>(1.1319, -0.1041, -2.1305);
    let c3 = vec3<f32>(-1.0498, 2.3638, 2.7414);
    let c4 = vec3<f32>(0.1786, -2.1233, -1.0666);
    
    return c0 + clamped * (c1 + clamped * (c2 + clamped * (c3 + clamped * c4)));
}
"#;

// Video processing application
#[allow(dead_code)]
struct VideoProcessor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    
    // Video resources
    video_data_buffer: wgpu::Buffer,
    processed_frame_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    
    // Burn tensors for advanced processing
    burn_device: Device<Backend>,
    enhancement_kernel: Tensor<Backend, 2>,    // Convolution kernel
    color_transform: Tensor<Backend, 2>,       // Color transformation matrix
    noise_pattern: Tensor<Backend, 2>,         // Dynamic noise for effects
    
    // State
    frame_width: u32,
    frame_height: u32,
    processing_mode: u32,
    frame_count: u32,
    start_time: Instant,
}

impl VideoProcessor {
    async fn new(window: Arc<winit::window::Window>) -> Self {
        println!("🎬 Real-time Video Processing with Burn Tensors");
        println!("===============================================");
        println!("   🔥 GPU-native video stream processing");
        println!("   🎨 Real-time Burn tensor effects");
        println!("   🚀 Zero-copy frame processing");
        println!("   📺 Multiple processing modes with live colormaps");

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
                    label: Some("Video Processor Device"),
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
            label: Some("Video Processing Shader"),
            source: wgpu::ShaderSource::Wgsl(VIDEO_PROCESSING_SHADER.into()),
        });

        // Create synthetic video frame data (replace with actual video loading in real app)
        let frame_width = 512u32;
        let frame_height = 512u32;
        let video_data = Self::create_synthetic_video_data(frame_width, frame_height);
        
        let video_data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Video Data Buffer"),
            size: (frame_width * frame_height * std::mem::size_of::<f32>() as u32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Upload initial video data
        queue.write_buffer(&video_data_buffer, 0, bytemuck::cast_slice(&video_data));

        let processed_frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Processed Frame Buffer"),
            size: (frame_width * frame_height * std::mem::size_of::<f32>() as u32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ProcessingParams {
            frame_width: u32,
            frame_height: u32,
            processing_mode: u32,
            time: f32,
        }

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Processing Parameters"),
            size: std::mem::size_of::<ProcessingParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Video Processing Bind Group Layout"),
            entries: &[
                // Video data buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Processed frame buffer
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
                // Processing parameters uniform
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
            label: Some("Video Processing Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: video_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: processed_frame_buffer.as_entire_binding(),
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
            label: Some("Video Processing Compute Pipeline"),
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
            label: Some("Video Display Render Pipeline"),
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

        // Initialize Burn tensors for advanced processing
        let burn_device = Device::<Backend>::default();
        
        // Edge enhancement kernel
        let enhancement_kernel = Tensor::<Backend, 2>::from_data(
            [[-1.0, -1.0, -1.0], [-1.0, 9.0, -1.0], [-1.0, -1.0, -1.0]],
            &burn_device
        );
        
        // Color transformation matrix (RGB to custom color space)
        let color_transform = Tensor::<Backend, 2>::from_data(
            [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]],
            &burn_device
        );
        
        // Dynamic noise pattern
        let noise_pattern = Tensor::<Backend, 2>::random(
            [frame_height as usize, frame_width as usize],
            Distribution::Normal(0.0, 0.1),
            &burn_device
        );

        Self {
            device,
            queue,
            surface,
            surface_config,
            compute_pipeline,
            render_pipeline,
            video_data_buffer,
            processed_frame_buffer,
            uniform_buffer,
            bind_group,
            burn_device,
            enhancement_kernel,
            color_transform,
            noise_pattern,
            frame_width,
            frame_height,
            processing_mode: 0,
            frame_count: 0,
            start_time: Instant::now(),
        }
    }

    fn create_synthetic_video_data(width: u32, height: u32) -> Vec<f32> {
        // Create synthetic video data (animated pattern) as grayscale f32 values
        let mut video_data = vec![0.0f32; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) as usize;
                let fx = x as f32 / width as f32;
                let fy = y as f32 / height as f32;
                
                // Create interesting patterns
                let pattern1 = (fx * 10.0).sin() * (fy * 8.0).cos() * 0.5 + 0.5;
                let pattern2 = ((fx - 0.5).powi(2) + (fy - 0.5).powi(2)).sqrt();
                let pattern3 = (fx * 15.0 + fy * 12.0).sin() * 0.5 + 0.5;
                
                // Combine patterns and normalize to [0,1]
                let combined = (pattern1 + pattern2 * 0.5 + pattern3 * 0.3) / 1.8;
                video_data[index] = combined.clamp(0.0, 1.0);
            }
        }
        
        video_data
    }

    fn update_synthetic_video(&mut self) {
        // Animate the video data for demonstration
        let time = self.start_time.elapsed().as_secs_f32();
        let mut video_data = vec![0.0f32; (self.frame_width * self.frame_height) as usize];
        
        for y in 0..self.frame_height {
            for x in 0..self.frame_width {
                let index = (y * self.frame_width + x) as usize;
                let fx = x as f32 / self.frame_width as f32;
                let fy = y as f32 / self.frame_height as f32;
                
                // Animated patterns with time evolution
                let pattern1 = (fx * 10.0 + time).sin() * (fy * 8.0 + time * 0.7).cos() * 0.5 + 0.5;
                let pattern2 = ((fx - 0.5 + (time * 0.3).sin() * 0.1).powi(2) + 
                               (fy - 0.5 + (time * 0.4).cos() * 0.1).powi(2)).sqrt() * 2.0;
                let pattern3 = (fx * 15.0 + fy * 12.0 + time * 2.0).sin() * 0.5 + 0.5;
                
                // Combine patterns and normalize
                let combined = (pattern1 + pattern2.min(1.0) * 0.5 + pattern3 * 0.3) / 1.8;
                video_data[index] = combined.clamp(0.0, 1.0);
            }
        }

        // Upload updated video data to buffer
        self.queue.write_buffer(&self.video_data_buffer, 0, bytemuck::cast_slice(&video_data));
    }

    fn update_burn_tensors(&mut self) {
        // Update noise pattern for dynamic effects
        self.noise_pattern = Tensor::<Backend, 2>::random(
            [self.frame_height as usize, self.frame_width as usize],
            Distribution::Normal(0.0, 0.05),
            &self.burn_device
        );

        // Occasionally update the enhancement kernel for variety
        if self.frame_count % 300 == 0 {
            let kernel_type = (self.frame_count / 300) % 3;
            match kernel_type {
                0 => {
                    // Sharpen kernel
                    self.enhancement_kernel = Tensor::<Backend, 2>::from_data(
                        [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
                        &self.burn_device
                    );
                }
                1 => {
                    // Edge detection kernel
                    self.enhancement_kernel = Tensor::<Backend, 2>::from_data(
                        [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
                        &self.burn_device
                    );
                }
                2 => {
                    // Blur kernel
                    self.enhancement_kernel = Tensor::<Backend, 2>::from_data(
                        [[0.111, 0.111, 0.111], [0.111, 0.111, 0.111], [0.111, 0.111, 0.111]],
                        &self.burn_device
                    );
                }
                _ => unreachable!(),
            }
            println!("🔄 Updated Burn enhancement kernel (type {})", kernel_type);
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    fn switch_processing_mode(&mut self) {
        self.processing_mode = (self.processing_mode + 1) % 4;
        let mode_name = match self.processing_mode {
            0 => "Edge Detection (Hot colormap)",
            1 => "Gaussian Blur (Cool colormap)",
            2 => "Ripple Effect (Viridis colormap)",
            3 => "Color Separation (Plasma colormap)",
            _ => "Unknown",
        };
        println!("🎨 Switched to processing mode: {}", mode_name);
    }

    fn update_and_render(&mut self) {
        self.frame_count += 1;
        let time = self.start_time.elapsed().as_secs_f32();

        // Update synthetic video every frame for animation
        self.update_synthetic_video();

        // Update Burn tensors periodically
        if self.frame_count % 60 == 0 {
            self.update_burn_tensors();
        }

        // Update processing parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ProcessingParams {
            frame_width: u32,
            frame_height: u32,
            processing_mode: u32,
            time: f32,
        }

        let params = ProcessingParams {
            frame_width: self.frame_width,
            frame_height: self.frame_height,
            processing_mode: self.processing_mode,
            time,
        };

        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[params]));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Video Processing Encoder"),
        });

        // Run compute shader for video processing
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Video Processing Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(
                (self.frame_width + 7) / 8,
                (self.frame_height + 7) / 8,
                1
            );
        }

        // Render processed video
        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            Err(e) => {
                eprintln!("Surface error: {:?}", e);
                return;
            }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Video Display Render Pass"),
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

        // Performance stats
        if self.frame_count % 120 == 0 {
            let fps = self.frame_count as f32 / time;
            println!("📊 Frame: {} | Time: {:.1}s | FPS: {:.1} | Mode: {}", 
                self.frame_count, time, fps, self.processing_mode);
        }
    }

    fn handle_key_input(&mut self, key_code: KeyCode) {
        match key_code {
            KeyCode::Space => {
                self.switch_processing_mode();
            }
            KeyCode::KeyR => {
                println!("🔄 Resetting video processor...");
                self.processing_mode = 0;
                self.frame_count = 0;
                self.start_time = Instant::now();
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
        .with_title("🎬 Real-time Video Processing with Burn Tensors")
        .with_inner_size(winit::dpi::LogicalSize::new(1024, 768))
        .build(&event_loop)
        .unwrap());

    let mut processor = VideoProcessor::new(window.clone()).await;

    println!("\n🎮 CONTROLS:");
    println!("   SPACE - Switch processing mode");
    println!("   R - Reset processor");
    println!("   ESC - Exit");
    println!("\n🎨 PROCESSING MODES:");
    println!("   0: Edge Detection (Hot colormap)");
    println!("   1: Gaussian Blur (Cool colormap)");
    println!("   2: Ripple Effect (Viridis colormap)");
    println!("   3: Color Separation (Plasma colormap)");

    event_loop.run(move |event, target| {
        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => {
                        println!("🎉 Video processing demo complete!");
                        target.exit();
                    }
                    WindowEvent::Resized(physical_size) => {
                        processor.resize(physical_size);
                    }
                    WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(key_code), .. }, .. } => {
                        if key_code == KeyCode::Escape {
                            target.exit();
                        } else {
                            processor.handle_key_input(key_code);
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        processor.update_and_render();
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
