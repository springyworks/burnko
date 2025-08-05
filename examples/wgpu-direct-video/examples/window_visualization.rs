//! Window-based tensor visualization example
//! 
//! This example demonstrates tensor visualization in a real window using winit + WGPU

use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    tensor::Tensor,
};
use std::time::Instant;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use futures::executor::block_on;

type Backend = Wgpu<f32>;

fn main() {
    println!("🪟 WGPU Direct Video Pipeline - Window Example");
    
    // Create event loop and window
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("WGPU Direct Video Pipeline")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
        .build(&event_loop)
        .unwrap();

    // Initialize WGPU with the window
    let renderer = block_on(WgpuRenderer::new(&window));
    println!("✅ WGPU renderer initialized");

    let mut app_state = AppState::new();
    
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    println!("🏁 Window closed");
                    elwt.exit();
                },
                WindowEvent::RedrawRequested => {
                    app_state.update();
                    
                    match renderer.render(&app_state) {
                        Ok(_) => {},
                        Err(wgpu::SurfaceError::Lost) => {
                            println!("⚠️ Surface lost, reconfiguring...");
                            // renderer.resize(renderer.size); // Would need to implement
                        },
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            println!("❌ Out of memory!");
                            elwt.exit();
                        },
                        Err(e) => eprintln!("⚠️ Render error: {:?}", e),
                    }
                },
                WindowEvent::Resized(physical_size) => {
                    println!("📐 Window resized to: {}x{}", physical_size.width, physical_size.height);
                    // renderer.resize(physical_size); // Would need to implement
                },
                _ => {}
            },
            Event::AboutToWait => {
                window.request_redraw();
            },
            _ => {}
        }
    }).unwrap();
}

/// Application state managing tensor generation and visualization
struct AppState {
    device: WgpuDevice,
    start_time: Instant,
    frame_count: u64,
    colormap_index: usize,
}

impl AppState {
    fn new() -> Self {
        Self {
            device: WgpuDevice::default(),
            start_time: Instant::now(),
            frame_count: 0,
            colormap_index: 0,
        }
    }
    
    fn update(&mut self) {
        self.frame_count += 1;
        
        // Change colormap every 180 frames (roughly every 3 seconds at 60 FPS)
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
    }
    
    fn generate_current_tensor(&self) -> Tensor<Backend, 2> {
        let t = self.start_time.elapsed().as_secs_f32();
        create_dynamic_pattern(t, &self.device)
    }
}

/// WGPU renderer for tensor visualization
struct WgpuRenderer {
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
}

impl WgpuRenderer {
    async fn new(window: &'static winit::window::Window) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        let surface = instance.create_surface(window).unwrap();
        
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.unwrap();
        
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: None,
            },
            None,
        ).await.unwrap();
        
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
            
        let size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        
        surface.configure(&device, &surface_config);
        
        Self {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
            surface,
            surface_config,
        }
    }
    
    fn render(&self, app_state: &AppState) -> Result<(), wgpu::SurfaceError> {
        let tensor = app_state.generate_current_tensor();
        let tensor_stats = analyze_tensor(&tensor);
        
        // For now, just clear the screen with a color based on tensor mean
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        // Use tensor mean as background color intensity
        let mean = tensor.clone().mean().into_scalar().min(1.0).max(0.0);
        let color_intensity = (mean * 0.5 + 0.2) as f64; // Scale to reasonable range
        
        let clear_color = match app_state.colormap_index {
            0 => wgpu::Color { r: color_intensity * 0.267, g: color_intensity * 0.004, b: color_intensity * 0.329, a: 1.0 }, // Viridis-like
            1 => wgpu::Color { r: color_intensity * 0.050, g: color_intensity * 0.029, b: color_intensity * 0.527, a: 1.0 }, // Plasma-like  
            2 => wgpu::Color { r: color_intensity, g: color_intensity * 0.5, b: 0.0, a: 1.0 }, // Hot-like
            3 => wgpu::Color { r: 0.0, g: color_intensity * 0.5, b: color_intensity, a: 1.0 }, // Cool-like
            _ => wgpu::Color { r: 0.0, g: color_intensity * 0.7, b: color_intensity, a: 1.0 }, // Jet-like
        };
        
        {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        // Print tensor stats occasionally for debugging
        if app_state.frame_count % 120 == 0 {
            println!("🔢 Tensor stats: {}", tensor_stats);
        }
        
        Ok(())
    }
}

/// Create a dynamic 2D pattern that evolves over time
fn create_dynamic_pattern(t: f32, device: &WgpuDevice) -> Tensor<Backend, 2> {
    let size = 256; // Larger for better visualization
    let mut data = Vec::with_capacity(size * size);
    
    // Generate complex dynamic pattern
    for y in 0..size {
        for x in 0..size {
            let x_norm = x as f32 / size as f32 * 8.0;
            let y_norm = y as f32 / size as f32 * 8.0;
            
            // Multiple interference patterns
            let wave1 = (x_norm + t * 2.0).sin() * 0.3;
            let wave2 = (y_norm + t * 1.5).sin() * 0.3;
            let wave3 = ((x_norm + y_norm) * 0.7 + t * 1.2).sin() * 0.2;
            let ripple = (((x_norm - 4.0).powi(2) + (y_norm - 4.0).powi(2)).sqrt() - t * 2.0).sin() * 0.2;
            
            // Add some noise for texture
            let noise = ((x_norm * 13.7 + y_norm * 17.3 + t * 3.1).sin() * 
                        (x_norm * 7.1 + y_norm * 11.9 + t * 2.7).cos()) * 0.1;
            
            let value = (wave1 + wave2 + wave3 + ripple + noise) * 0.5 + 0.5;
            data.push(value.clamp(0.0, 1.0));
        }
    }
    
    Tensor::<Backend, 1>::from_floats(data.as_slice(), device).reshape([size, size])
}

/// Analyze tensor for interesting statistics
fn analyze_tensor(tensor: &Tensor<Backend, 2>) -> String {
    let mean = tensor.clone().mean().into_scalar();
    let min_val = tensor.clone().min().into_scalar();
    let max_val = tensor.clone().max().into_scalar();
    
    format!("mean: {:.3}, range: [{:.3}, {:.3}]", mean, min_val, max_val)
}
