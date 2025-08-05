//! Basic tensor streaming example
//! 
//! This example demonstrates the core functionality of the WGPU Direct Video Pipeline:
//! - Creating tensors on GPU
//! - Converting them to textures without CPU transfers
//! - Streaming them in real-time with different colormaps

use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    tensor::Tensor,
};
use burn_wgpu::video::{
    VideoStream, VideoStreamBuilder, VideoConfig, ColorMap, TensorVideoStream
};
use std::time::{Duration, Instant};

type Backend = Wgpu<f32>;

#[tokio::main]
async fn main() {
    println!("🎥 WGPU Direct Video Pipeline - Basic Tensor Streaming");
    
    // Initialize WGPU device
    let device = WgpuDevice::default();
    println!("✅ WGPU device initialized: {:?}", device);
    
    // Create a video stream for tensor visualization
    let config = VideoConfig::debug(); // High FPS for debugging
    let mut video_stream = VideoStreamBuilder::new()
        .dimensions(512, 512)
        .fps(60)
        .build();
    
    println!("✅ Video stream created with config: {:?}", config);
    
    // Create some dynamic tensor data
    println!("🔄 Generating dynamic tensor patterns...");
    
    for frame in 0..1000 {
        let start_time = Instant::now();
        
        // Create a dynamic tensor pattern that evolves over time
        let t = frame as f32 * 0.1;
        let tensor = create_dynamic_pattern(t, &device);
        
        // Push tensor to video stream (this should be GPU-to-GPU)
        match video_stream.push_tensor_frame(&tensor) {
            Ok(_) => {
                if frame % 60 == 0 { // Log every second at 60fps
                    let metrics = video_stream.metrics();
                    println!("📊 Frame {}: {} rendered, {} dropped, avg {:.2}ms", 
                             frame, metrics.frames_rendered, metrics.frames_dropped,
                             metrics.avg_frame_time.as_millis());
                }
            }
            Err(e) => {
                eprintln!("❌ Failed to push frame {}: {}", frame, e);
                break;
            }
        }
        
        // Change colormap every 100 frames
        if frame % 100 == 0 && frame > 0 {
            let colormap = match (frame / 100) % 5 {
                0 => ColorMap::Viridis,
                1 => ColorMap::Plasma,
                2 => ColorMap::Hot,
                3 => ColorMap::Cool,
                _ => ColorMap::Jet,
            };
            video_stream.set_colormap(colormap);
            println!("🎨 Changed colormap to: {:?}", colormap);
        }
        
        // Render frame to output
        if let Err(e) = video_stream.render_frame() {
            eprintln!("❌ Failed to render frame {}: {}", frame, e);
        }
        
        // Control frame rate
        let frame_time = start_time.elapsed();
        let target_frame_time = Duration::from_millis(1000 / config.fps as u64);
        if frame_time < target_frame_time {
            tokio::time::sleep(target_frame_time - frame_time).await;
        }
    }
    
    let final_metrics = video_stream.metrics();
    println!("🏁 Final metrics:");
    println!("   Total frames rendered: {}", final_metrics.frames_rendered);
    println!("   Total frames dropped: {}", final_metrics.frames_dropped);
    println!("   Average frame time: {:.2}ms", final_metrics.avg_frame_time.as_millis());
    println!("   Frame drop rate: {:.1}%", 
             final_metrics.frames_dropped as f32 / final_metrics.frames_rendered as f32 * 100.0);
}

/// Create a dynamic 2D pattern that evolves over time
fn create_dynamic_pattern(t: f32, device: &WgpuDevice) -> Tensor<Backend, 2> {
    let size = 256;
    let mut data = Vec::with_capacity(size * size);
    
    // Generate a dynamic pattern with multiple frequency components
    for y in 0..size {
        for x in 0..size {
            let x_norm = x as f32 / size as f32 * 8.0;
            let y_norm = y as f32 / size as f32 * 8.0;
            
            // Multiple sine waves with time evolution
            let wave1 = (x_norm + t).sin() * 0.3;
            let wave2 = (y_norm + t * 1.5).sin() * 0.3;
            let wave3 = ((x_norm + y_norm) * 0.5 + t * 0.8).sin() * 0.2;
            let ripple = (((x_norm - 4.0).powi(2) + (y_norm - 4.0).powi(2)).sqrt() - t * 2.0).sin() * 0.2;
            
            let value = (wave1 + wave2 + wave3 + ripple) * 0.5 + 0.5;
            data.push(value);
        }
    }
    
    Tensor::from_floats(data.as_slice(), device).reshape([size, size])
}

/// Create a rotating pattern
fn create_rotating_pattern(t: f32, device: &WgpuDevice) -> Tensor<Backend, 2> {
    let size = 256;
    let mut data = Vec::with_capacity(size * size);
    let center = size as f32 / 2.0;
    
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let angle = dy.atan2(dx) + t;
            let radius = (dx * dx + dy * dy).sqrt() / center;
            
            let value = (angle * 4.0).sin() * (1.0 - radius).max(0.0);
            data.push((value + 1.0) * 0.5);
        }
    }
    
    Tensor::from_floats(data.as_slice(), device).reshape([size, size])
}

/// Create a cellular automata-like pattern
fn create_cellular_pattern(t: f32, device: &WgpuDevice) -> Tensor<Backend, 2> {
    let size = 128;
    let mut data = Vec::with_capacity(size * size);
    
    for y in 0..size {
        for x in 0..size {
            let x_norm = x as f32 / size as f32;
            let y_norm = y as f32 / size as f32;
            
            // Create cellular-like patterns
            let cell1 = ((x_norm * 16.0 + t).sin() + (y_norm * 16.0 + t).cos()).abs();
            let cell2 = ((x_norm * 8.0 - t * 0.7).cos() * (y_norm * 8.0 - t * 0.7).sin()).abs();
            
            let value = (cell1 + cell2) * 0.25;
            data.push(value.clamp(0.0, 1.0));
        }
    }
    
    Tensor::from_floats(data.as_slice(), device).reshape([size, size])
}
