//! Video configuration and settings for tensor streaming

use std::time::Duration;

/// Configuration for video streaming
#[derive(Debug, Clone)]
pub struct VideoConfig {
    /// Output video width in pixels
    pub width: u32,
    /// Output video height in pixels
    pub height: u32,
    /// Frames per second
    pub fps: u32,
    /// Video format for output
    pub format: VideoFormat,
    /// Output destination
    pub output: VideoOutput,
    /// Buffer size for frame history
    pub buffer_size: usize,
    /// Maximum frame drop threshold
    pub max_frame_drop: f32,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            fps: 30,
            format: VideoFormat::Rgba8,
            output: VideoOutput::Window,
            buffer_size: 3, // Triple buffering
            max_frame_drop: 0.1, // Drop max 10% of frames under load
        }
    }
}

impl VideoConfig {
    /// Create configuration optimized for training visualization
    pub fn training() -> Self {
        Self {
            width: 256,
            height: 256,
            fps: 10, // Lower FPS for training
            format: VideoFormat::Rgba8,
            output: VideoOutput::Window,
            buffer_size: 2,
            max_frame_drop: 0.2,
        }
    }
    
    /// Create configuration optimized for debugging
    pub fn debug() -> Self {
        Self {
            width: 512,
            height: 512,
            fps: 60, // High FPS for debugging
            format: VideoFormat::Rgba8,
            output: VideoOutput::Window,
            buffer_size: 5,
            max_frame_drop: 0.0, // No frame dropping for debugging
        }
    }
    
    /// Create configuration for high-quality export
    pub fn export() -> Self {
        Self {
            width: 1024,
            height: 1024,
            fps: 30,
            format: VideoFormat::Rgba16,
            output: VideoOutput::File("tensor_video.mp4".to_string()),
            buffer_size: 10,
            max_frame_drop: 0.0,
        }
    }
}

/// Video pixel format
#[derive(Debug, Clone, Copy)]
pub enum VideoFormat {
    /// 8-bit RGBA
    Rgba8,
    /// 16-bit RGBA (higher precision)
    Rgba16,
    /// 32-bit RGBA (float precision)
    Rgba32Float,
}

impl VideoFormat {
    /// Get the bytes per pixel for this format
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            VideoFormat::Rgba8 => 4,
            VideoFormat::Rgba16 => 8,
            VideoFormat::Rgba32Float => 16,
        }
    }
    
    /// Get the corresponding WGPU texture format
    pub fn to_wgpu_format(&self) -> wgpu::TextureFormat {
        match self {
            VideoFormat::Rgba8 => wgpu::TextureFormat::Rgba8Unorm,
            VideoFormat::Rgba16 => wgpu::TextureFormat::Rgba16Unorm,
            VideoFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
        }
    }
}

/// Video output destination
#[derive(Debug, Clone)]
pub enum VideoOutput {
    /// Display in a window
    Window,
    /// Save to file
    File(String),
    /// Stream over network (future feature)
    Network { 
        address: String, 
        port: u16 
    },
    /// Multiple outputs
    Multiple(Vec<VideoOutput>),
}

/// Color mapping options for tensor visualization
#[derive(Debug, Clone, Copy)]
pub enum ColorMap {
    /// Grayscale mapping
    Grayscale,
    /// Viridis color scale (perceptually uniform)
    Viridis,
    /// Plasma color scale (high contrast)
    Plasma,
    /// Hot color scale (black-red-yellow-white)
    Hot,
    /// Cool color scale (cyan-blue-magenta)
    Cool,
    /// Jet color scale (blue-cyan-yellow-red)
    Jet,
    /// Custom RGB mapping
    Custom { 
        min_color: [f32; 3], 
        max_color: [f32; 3] 
    },
}

impl Default for ColorMap {
    fn default() -> Self {
        ColorMap::Viridis
    }
}

impl ColorMap {
    /// Get the shader code for this colormap
    pub fn shader_code(&self) -> &'static str {
        match self {
            ColorMap::Grayscale => "vec3(value, value, value)",
            ColorMap::Viridis => include_str!("shaders/viridis.wgsl"),
            ColorMap::Plasma => include_str!("shaders/plasma.wgsl"), 
            ColorMap::Hot => include_str!("shaders/hot.wgsl"),
            ColorMap::Cool => include_str!("shaders/cool.wgsl"),
            ColorMap::Jet => include_str!("shaders/jet.wgsl"),
            ColorMap::Custom { .. } => "mix(min_color, max_color, value)",
        }
    }
}

/// Performance tracking metrics
#[derive(Debug, Clone)]
pub struct VideoMetrics {
    /// Number of frames produced
    pub frames_produced: u64,
    /// Number of frames dropped due to performance
    pub frames_dropped: u64,
    /// Average frame time in milliseconds
    pub avg_frame_time_ms: f32,
    /// Current frames per second
    pub current_fps: f32,
    /// Last update timestamp
    pub last_update: std::time::Instant,
}

impl Default for VideoMetrics {
    fn default() -> Self {
        Self {
            frames_produced: 0,
            frames_dropped: 0,
            avg_frame_time_ms: 16.67, // 60 FPS target
            current_fps: 0.0,
            last_update: std::time::Instant::now(),
        }
    }
}
