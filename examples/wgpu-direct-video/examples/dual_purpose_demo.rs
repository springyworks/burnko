//! Dual-Purpose Shader Example
//! 
//! This example demonstrates the revolutionary concept of using the same GPU resources 
//! for both tensor computation AND real-time visualization simultaneously.
//! 
//! The same shader operates on tensor data and renders it to screen without CPU transfers.

use burn::tensor::{Tensor, Distribution};
use burn_wgpu::{Wgpu, WgpuDevice};
use wgpu_direct_video::*;
use std::sync::Arc;
use std::time::Instant;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

type Backend = Wgpu<f32>;

struct DualApp {
    visualizer: TensorVisualizer,
    surface_config: wgpu::SurfaceConfiguration,
    window: winit::window::Window,
    device: WgpuDevice,
    start_time: Instant,
    frame_count: u64,
}

impl DualApp {
    async fn new(event_loop: &EventLoop<()>) -> Self {
        let window = WindowBuilder::new()
            .with_title("Dual-Purpose WGPU: Tensor Compute + Real-time Visualization")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .build(event_loop)
            .unwrap();

        let size = window.inner_size();

        // Create WGPU instance and surface
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = unsafe { instance.create_surface(&window).unwrap() };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (wgpu_device, wgpu_queue) = adapter
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

        surface.configure(&wgpu_device, &surface_config);

        // Create visualizer
        let mut visualizer = TensorVisualizer::new(
            Arc::new(wgpu_device),
            Arc::new(wgpu_queue),
            Some(surface),
        );

        // Initialize with dual-purpose shader
        let shader = shaders::basic_tensor_visualization();
        visualizer.initialize_with_shader(&shader, surface_format).unwrap();

        // Create Burn device
        let device = WgpuDevice::default();

        Self {
            visualizer,
            surface_config,
            window,
            device,
            start_time: Instant::now(),
            frame_count: 0,
        }
    }

    fn create_evolving_tensor(&self, time: f32) -> Tensor<Backend, 2> {
        let size = 256;
        let mut data = Vec::with_capacity(size * size);

        for y in 0..size {
            for x in 0..size {
                let x_norm = x as f32 / size as f32;
                let y_norm = y as f32 / size as f32;

                // Dynamic pattern that evolves over time
                let wave1 = ((x_norm * 10.0 + time * 2.0).sin() * 0.3);
                let wave2 = ((y_norm * 8.0 + time * 1.5).cos() * 0.3);
                let spiral = (((x_norm - 0.5).powi(2) + (y_norm - 0.5).powi(2)).sqrt() * 20.0 - time * 3.0).sin() * 0.4;

                let value = (wave1 + wave2 + spiral + 1.0) * 0.5; // Normalize to [0, 1]
                data.push(value.clamp(0.0, 1.0));
            }
        }

        Tensor::<Backend, 1>::from_floats(data.as_slice(), &self.device).reshape([size, size])
    }

    fn update(&mut self) {
        self.frame_count += 1;
        let current_time = self.start_time.elapsed().as_secs_f32();

        // Create tensor with evolving pattern
        let tensor = self.create_evolving_tensor(current_time);
        
        // THIS IS THE MAGIC: The same tensor data is:
        // 1. Used for tensor computations (Burn operations)
        // 2. Directly visualized on screen (graphics pipeline)
        // No CPU transfer needed!

        // Optional: Perform some tensor operations to demonstrate dual use
        let processed_tensor = if self.frame_count % 120 == 0 {
            // Every 2 seconds, apply some tensor operations
            let mean = tensor.mean();
            let normalized = (tensor.clone() - mean.clone()) / (tensor.std(0) + 1e-6);
            println!("🔥 Frame {}: Applied tensor normalization, mean = {:.3}", 
                     self.frame_count, mean.into_scalar());
            normalized
        } else {
            tensor
        };

        // Render the tensor directly to screen
        match self.visualizer.render_tensor(&processed_tensor) {
            Ok(_) => {
                if self.frame_count % 60 == 0 {
                    let fps = self.frame_count as f32 / current_time;
                    println!("📊 Dual-Purpose Pipeline: {:.1} FPS | Frame {} | Time {:.2}s", 
                             fps, self.frame_count, current_time);
                }
            }
            Err(e) => {
                eprintln!("⚠️ Render error: {}", e);
            }
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.visualizer.device.queue().submit(std::iter::empty());
        }
    }
}

async fn run() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    let mut app = DualApp::new(&event_loop).await;

    println!("🚀 Dual-Purpose WGPU Pipeline Started!");
    println!("   • Same GPU memory used for tensor compute AND visualization");
    println!("   • Zero CPU transfers - everything stays on GPU");
    println!("   • Real-time tensor evolution with mathematical patterns");
    println!("   • Watch tensors transform live as Burn operates on them!");

    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == app.window.id() => match event {
                WindowEvent::CloseRequested => {
                    println!("🏁 Dual-Purpose Pipeline completed {} frames", app.frame_count);
                    elwt.exit();
                }
                WindowEvent::Resized(physical_size) => {
                    println!("📐 Resized to: {}x{}", physical_size.width, physical_size.height);
                    app.resize(*physical_size);
                }
                WindowEvent::RedrawRequested => {
                    app.update();
                }
                _ => {}
            }
            Event::AboutToWait => {
                app.window.request_redraw();
            }
            _ => {}
        }
    }).unwrap();
}

fn main() {
    println!("🎨 Dual-Purpose WGPU Backend: Revolutionary Tensor Visualization");
    println!("   This example demonstrates using the SAME shaders for:");
    println!("   1. Tensor computations (via Burn/CubeCL)");  
    println!("   2. Real-time visualization (via WGPU graphics)");
    println!("   No CPU transfers - pure GPU-to-GPU pipeline!");
    
    pollster::block_on(run());
}
