//! Simple window-based tensor visualization example
//! 
//! This example demonstrates tensor visualization in a window using winit + WGPU
//! without complex lifetime issues

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

type Backend = Wgpu<f32>;

fn main() {
    println!("🪟 WGPU Direct Video Pipeline - Simple Window Example");
    
    // Create event loop and window
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("WGPU Direct Video Pipeline")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
        .build(&event_loop)
        .unwrap();

    println!("✅ Window created");

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
                    app_state.render();
                },
                WindowEvent::Resized(physical_size) => {
                    println!("📐 Window resized to: {}x{}", physical_size.width, physical_size.height);
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
        println!("🚀 Initializing WGPU device...");
        let device = WgpuDevice::default();
        println!("✅ WGPU device initialized");
        
        Self {
            device,
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
    
    fn render(&self) {
        // Generate tensor and analyze it
        let tensor = self.generate_current_tensor();
        let tensor_stats = analyze_tensor(&tensor);
        
        // Print tensor stats occasionally for debugging
        if self.frame_count % 120 == 0 {
            println!("🔢 Tensor stats: {}", tensor_stats);
        }
        
        // In a real implementation, this is where we would:
        // 1. Convert the tensor to a texture on GPU
        // 2. Apply the current colormap using a fragment shader
        // 3. Render the textured quad to the window surface
        // 4. Present the frame
        
        // For now, we're just demonstrating the tensor generation pipeline
    }
    
    fn generate_current_tensor(&self) -> Tensor<Backend, 2> {
        let t = self.start_time.elapsed().as_secs_f32();
        create_dynamic_pattern(t, &self.device)
    }
}

/// Create a dynamic 2D pattern that evolves over time
fn create_dynamic_pattern(t: f32, device: &WgpuDevice) -> Tensor<Backend, 2> {
    let size = 256; // Good size for visualization
    let mut data = Vec::with_capacity(size * size);
    
    // Generate complex dynamic pattern with multiple wave components
    for y in 0..size {
        for x in 0..size {
            let x_norm = x as f32 / size as f32 * 8.0;
            let y_norm = y as f32 / size as f32 * 8.0;
            
            // Multiple interference patterns
            let wave1 = (x_norm + t * 2.0).sin() * 0.3;
            let wave2 = (y_norm + t * 1.5).sin() * 0.3;
            let wave3 = ((x_norm + y_norm) * 0.7 + t * 1.2).sin() * 0.2;
            let ripple = (((x_norm - 4.0).powi(2) + (y_norm - 4.0).powi(2)).sqrt() - t * 2.0).sin() * 0.2;
            
            // Add some high-frequency noise for texture
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
