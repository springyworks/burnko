//! Psychedelic Real-time Tensor Visualization 🌈✨
//! 
//! This example creates mesmerizing, dynamic visualizations of tensor operations
//! with psychedelic animations and real-time effects. Multiple visualization modes
//! showcase different tensor operations in beautiful, animated patterns.
//! 
//! Features:
//! - 🌈 Multiple psychedelic visualization modes
//! - ✨ Real-time tensor operation animations  
//! - 🎭 Dynamic pattern generation with time-based effects
//! - 🚀 CPU vs GPU performance comparison
//! - 🎨 Beautiful color gradients and effects
//! - 🔄 Smooth transitions between different operations

use burn_tensor::{Tensor, TensorData, Shape, ops::ScanOp};
use burn_ndarray::NdArrayBackend as CpuBackend;
use burn_wgpu::WgpuBackend as GpuBackend;
use burn_core::backend::Backend;
use minifb::{Key, Window, WindowOptions, Scale};
use std::time::{Duration, Instant};
use std::thread;
use std::sync::{Arc, Mutex};

type CpuDevice = <CpuBackend as Backend>::Device;
type GpuDevice = <GpuBackend as Backend>::Device;

type CpuBackend = burn_ndarray::NdArrayBackend<f32>;
type GpuBackend = burn_wgpu::WgpuBackend<f32>;

// Window and visualization constants
const WINDOW_WIDTH: usize = 1200;
const WINDOW_HEIGHT: usize = 800;
const TENSOR_WIDTH: usize = WINDOW_WIDTH / 4;  // Scale down for performance
const TENSOR_HEIGHT: usize = WINDOW_HEIGHT / 4;

#[derive(Clone, Copy, Debug)]
enum VisualizationMode {
    PsychedelicWaves,
    CosmicSpiral,
    TensorStorm, 
    PlasmaField,
    QuantumRipples,
    HypnoticMandalas,
}

impl VisualizationMode {
    fn next(self) -> Self {
        match self {
            Self::PsychedelicWaves => Self::CosmicSpiral,
            Self::CosmicSpiral => Self::TensorStorm,
            Self::TensorStorm => Self::PlasmaField,
            Self::PlasmaField => Self::QuantumRipples,
            Self::QuantumRipples => Self::HypnoticMandalas,
            Self::HypnoticMandalas => Self::PsychedelicWaves,
        }
    }
    
    fn name(self) -> &'static str {
        match self {
            Self::PsychedelicWaves => "Psychedelic Waves",
            Self::CosmicSpiral => "Cosmic Spiral",
            Self::TensorStorm => "Tensor Storm",
            Self::PlasmaField => "Plasma Field", 
            Self::QuantumRipples => "Quantum Ripples",
            Self::HypnoticMandalas => "Hypnotic Mandalas",
        }
    }
}

struct PsychedelicTensorVisualizer {
    window: Window,
    buffer: Vec<u32>,
    cpu_device: <CpuBackend as burn_tensor::backend::Backend>::Device,
    gpu_device: <GpuBackend as burn_tensor::backend::Backend>::Device,
    
    // Animation state
    time: f32,
    animation_speed: f32,
    visualization_mode: VisualizationMode,
    
    // Tensor operations
    current_operation: TensorOperation,
    use_gpu: bool,
    auto_cycle_operations: bool,
    auto_cycle_modes: bool,
    
    // Visual effects
    color_phase: f32,
    intensity_multiplier: f32,
    wave_complexity: f32,
    
    performance_stats: Arc<Mutex<PerformanceStats>>,
}

#[derive(Clone, Copy, Debug)]
enum TensorOperation {
    Sum,
    Max,
    Min,
    Cumsum,
}

impl TensorOperation {
    fn name(self) -> &'static str {
        match self {
            TensorOperation::Sum => "Sum",
            TensorOperation::Max => "Max",
            TensorOperation::Min => "Min",
            TensorOperation::Cumsum => "Cumsum",
        }
    }
    
    fn next(self) -> Self {
        match self {
            TensorOperation::Sum => TensorOperation::Max,
            TensorOperation::Max => TensorOperation::Min,
            TensorOperation::Min => TensorOperation::Cumsum,
            TensorOperation::Cumsum => TensorOperation::Sum,
        }
    }
}

#[derive(Clone)]
struct PerformanceStats {
    times: Vec<Duration>,
}

impl PerformanceStats {
    fn new() -> Self {
        Self {
            times: Vec::new(),
        }
    }
    
    fn add_timing(&mut self, duration: Duration) {
        self.times.push(duration);
        if self.times.len() > 100 {
            self.times.remove(0);
        }
    }
    
    fn average_time(&self) -> Option<Duration> {
        if self.times.is_empty() {
            None
        } else {
            let total: Duration = self.times.iter().sum();
            Some(total / self.times.len() as u32)
        }
    }
}

impl PsychedelicTensorVisualizer {
    fn new() -> Self {
        let mut window = Window::new(
            "🔥 Burn Psychedelic Tensor Visualizer 🌀",
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            WindowOptions {
                resize: false,
                scale: Scale::X1,
                ..WindowOptions::default()
            },
        ).expect("Failed to create window");
        
        window.limit_update_rate(Some(std::time::Duration::from_micros(16600))); // ~60fps
        
        let buffer = vec![0; WINDOW_WIDTH * WINDOW_HEIGHT];
        let cpu_device = CpuBackend::default_device();
        let gpu_device = GpuBackend::default_device();
        
        Self {
            window,
            buffer,
            cpu_device,
            gpu_device,
            time: 0.0,
            animation_speed: 1.0,
            visualization_mode: VisualizationMode::PsychedelicWaves,
            current_operation: TensorOperation::Sum,
            use_gpu: true,
            auto_cycle_operations: true,
            auto_cycle_modes: false,
            color_phase: 0.0,
            intensity_multiplier: 1.0,
            wave_complexity: 1.0,
            performance_stats: Arc::new(Mutex::new(PerformanceStats::new())),
        }
    }
    
    fn update_animation_state(&mut self) {
        self.time += 0.016 * self.animation_speed; // Assuming ~60fps
        self.color_phase += 0.02;
        
        // Auto-cycle operations every 5 seconds
        if self.auto_cycle_operations && (self.time as u32) % 5 == 0 && (self.time as u32) != ((self.time - 0.016) as u32) {
            self.current_operation = self.current_operation.next();
        }
        
        // Auto-cycle visualization modes every 15 seconds
        if self.auto_cycle_modes && (self.time as u32) % 15 == 0 && (self.time as u32) != ((self.time - 0.016) as u32) {
            self.visualization_mode = self.visualization_mode.next();
        }
    }
    
    fn generate_psychedelic_tensor(&self, width: usize, height: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(width * height);
        
        match self.visualization_mode {
            VisualizationMode::PsychedelicWaves => {
                for y in 0..height {
                    for x in 0..width {
                        let nx = x as f32 / width as f32;
                        let ny = y as f32 / height as f32;
                        
                        let wave1 = (nx * 20.0 + self.time * 2.0).sin();
                        let wave2 = (ny * 15.0 + self.time * 1.5).sin();
                        let wave3 = ((nx + ny) * 12.0 + self.time * 3.0).cos();
                        let interference = (nx * ny * 30.0 + self.time * 2.5).sin();
                        
                        let value = (wave1 + wave2 + wave3 + interference) * 0.25 * self.wave_complexity;
                        data.push(value);
                    }
                }
            },
            VisualizationMode::CosmicSpiral => {
                let center_x = width as f32 / 2.0;
                let center_y = height as f32 / 2.0;
                
                for y in 0..height {
                    for x in 0..width {
                        let dx = x as f32 - center_x;
                        let dy = y as f32 - center_y;
                        let distance = (dx * dx + dy * dy).sqrt();
                        let angle = dy.atan2(dx);
                        
                        let spiral = (distance * 0.1 - angle * 3.0 + self.time * 4.0).sin();
                        let radial = (distance * 0.05 + self.time * 2.0).cos();
                        let angular = (angle * 8.0 + self.time * 3.0).sin();
                        
                        let value = (spiral + radial + angular) / 3.0;
                        data.push(value);
                    }
                }
            },
            VisualizationMode::TensorStorm => {
                for y in 0..height {
                    for x in 0..width {
                        let nx = (x as f32 / width as f32) * 4.0;
                        let ny = (y as f32 / height as f32) * 4.0;
                        
                        let noise1 = (nx + self.time * 0.8).sin() * (ny + self.time * 1.2).cos();
                        let noise2 = ((nx * 2.0 + self.time * 1.5).sin() + (ny * 1.5 + self.time * 2.0).cos()) * 0.5;
                        let turbulence = (nx * ny + self.time * 3.0).sin() * 0.3;
                        
                        let storm = noise1 + noise2 + turbulence;
                        data.push(storm * self.intensity_multiplier);
                    }
                }
            },
            VisualizationMode::PlasmaField => {
                for y in 0..height {
                    for x in 0..width {
                        let nx = x as f32 / 32.0;
                        let ny = y as f32 / 32.0;
                        
                        let plasma1 = (nx + self.time).sin();
                        let plasma2 = (ny + self.time * 1.3).sin();
                        let plasma3 = ((nx + ny + self.time * 0.7) / 2.0).sin();
                        let plasma4 = (((nx - ny) * (nx - ny) + self.time * 2.0).sqrt()).sin();
                        
                        let value = (plasma1 + plasma2 + plasma3 + plasma4) * 0.25;
                        data.push(value);
                    }
                }
            },
            VisualizationMode::QuantumRipples => {
                let center_x = width as f32 / 2.0;
                let center_y = height as f32 / 2.0;
                
                for y in 0..height {
                    for x in 0..width {
                        let dx = x as f32 - center_x;
                        let dy = y as f32 - center_y;
                        let distance = (dx * dx + dy * dy).sqrt();
                        
                        let ripple1 = (distance * 0.2 - self.time * 8.0).sin();
                        let ripple2 = (distance * 0.15 - self.time * 6.0).cos();
                        let ripple3 = (distance * 0.1 - self.time * 10.0).sin();
                        let quantum = (distance * dx * dy * 0.001 + self.time * 5.0).sin();
                        
                        let value = (ripple1 + ripple2 + ripple3 + quantum) * 0.25;
                        data.push(value);
                    }
                }
            },
            VisualizationMode::HypnoticMandalas => {
                let center_x = width as f32 / 2.0;
                let center_y = height as f32 / 2.0;
                
                for y in 0..height {
                    for x in 0..width {
                        let dx = x as f32 - center_x;
                        let dy = y as f32 - center_y;
                        let distance = (dx * dx + dy * dy).sqrt();
                        let angle = dy.atan2(dx);
                        
                        let mandala1 = (angle * 8.0 + distance * 0.1 + self.time * 2.0).sin();
                        let mandala2 = (angle * 12.0 - distance * 0.05 + self.time * 1.5).cos();
                        let mandala3 = (angle * 16.0 + distance * 0.08 - self.time * 3.0).sin();
                        let radial_mod = (distance * 0.2 + self.time).sin();
                        
                        let value = (mandala1 + mandala2 + mandala3) * radial_mod * 0.33;
                        data.push(value);
                    }
                }
            },
        }
        
        data
    }

    fn apply_tensor_operation(&self, data: Vec<f32>) -> Vec<f32> {
        let start = std::time::Instant::now();
        
        let result = if self.use_gpu {
            self.apply_gpu_operation(data)
        } else {
            self.apply_cpu_operation(data)
        };
        
        let duration = start.elapsed();
        if let Ok(mut stats) = self.performance_stats.try_lock() {
            stats.add_timing(duration);
        }
        
        result
    }
    
    fn apply_gpu_operation(&self, data: Vec<f32>) -> Vec<f32> {
        let tensor: burn_tensor::Tensor<GpuBackend, 2> = 
            burn_tensor::Tensor::from_floats(data.as_slice(), &self.gpu_device)
                .reshape([TENSOR_HEIGHT, TENSOR_WIDTH]);
        
        let result_tensor = match self.current_operation {
            TensorOperation::Sum => tensor.sum_dim(1),
            TensorOperation::Max => tensor.max_dim(1),
            TensorOperation::Min => tensor.min_dim(1),
            TensorOperation::Cumsum => {
                // Create a cumulative sum along dimension 1
                let mut result = tensor.clone();
                for i in 1..TENSOR_WIDTH {
                    let prev_slice = result.clone().slice([0..TENSOR_HEIGHT, (i-1)..i]);
                    let curr_slice = result.clone().slice([0..TENSOR_HEIGHT, i..(i+1)]);
                    let new_slice = prev_slice + curr_slice;
                    result = result.slice_assign([0..TENSOR_HEIGHT, i..(i+1)], new_slice);
                }
                result.sum_dim(1)
            },
        };
        
        // Convert back to full 2D for visualization
        let result_data: Vec<f32> = result_tensor.to_data().as_slice().iter().cloned().collect();
        self.expand_to_full_size(result_data)
    }
    
    fn apply_cpu_operation(&self, data: Vec<f32>) -> Vec<f32> {
        let tensor: burn_tensor::Tensor<CpuBackend, 2> = 
            burn_tensor::Tensor::from_floats(data.as_slice(), &self.cpu_device)
                .reshape([TENSOR_HEIGHT, TENSOR_WIDTH]);
        
        let result_tensor = match self.current_operation {
            TensorOperation::Sum => tensor.sum_dim(1),
            TensorOperation::Max => tensor.max_dim(1),
            TensorOperation::Min => tensor.min_dim(1),
            TensorOperation::Cumsum => {
                let mut result = tensor.clone();
                for i in 1..TENSOR_WIDTH {
                    let prev_slice = result.clone().slice([0..TENSOR_HEIGHT, (i-1)..i]);
                    let curr_slice = result.clone().slice([0..TENSOR_HEIGHT, i..(i+1)]);
                    let new_slice = prev_slice + curr_slice;
                    result = result.slice_assign([0..TENSOR_HEIGHT, i..(i+1)], new_slice);
                }
                result.sum_dim(1)
            },
        };
        
        let result_data: Vec<f32> = result_tensor.to_data().as_slice().iter().cloned().collect();
        self.expand_to_full_size(result_data)
    }
    
    fn expand_to_full_size(&self, data: Vec<f32>) -> Vec<f32> {
        // Expand the reduced tensor back to full visualization size with interpolation
        let mut full_data = vec![0.0; TENSOR_HEIGHT * TENSOR_WIDTH];
        
        for y in 0..TENSOR_HEIGHT {
            let value = if y < data.len() { data[y] } else { 0.0 };
            for x in 0..TENSOR_WIDTH {
                full_data[y * TENSOR_WIDTH + x] = value;
            }
        }
        
        full_data
    }
    
    fn tensor_to_color(&self, value: f32) -> u32 {
        // Psychedelic color mapping with time-based phase shifting
        let normalized = (value.tanh() + 1.0) * 0.5; // Normalize to [0, 1]
        
        let phase_r = self.color_phase;
        let phase_g = self.color_phase + 2.0943; // 120 degrees
        let phase_b = self.color_phase + 4.1888; // 240 degrees
        
        let r = ((normalized * 6.28 + phase_r).sin() * 0.5 + 0.5) * 255.0;
        let g = ((normalized * 6.28 + phase_g).sin() * 0.5 + 0.5) * 255.0;
        let b = ((normalized * 6.28 + phase_b).sin() * 0.5 + 0.5) * 255.0;
        
        // Add intensity modulation
        let intensity = self.intensity_multiplier * (1.0 + (self.time * 4.0).sin() * 0.2);
        
        let r = (r * intensity).min(255.0) as u32;
        let g = (g * intensity).min(255.0) as u32;
        let b = (b * intensity).min(255.0) as u32;
        
        (255 << 24) | (r << 16) | (g << 8) | b
    }

    fn render(&mut self) {
        // Update animation state
        self.update_animation_state();
        
        // Generate psychedelic tensor pattern
        let tensor_data = self.generate_psychedelic_tensor(TENSOR_WIDTH, TENSOR_HEIGHT);
        
        // Apply tensor operations for additional effects
        let processed_data = self.apply_tensor_operation(tensor_data);
        
        // Convert to colors and update buffer
        for (i, &value) in processed_data.iter().enumerate() {
            if i < self.buffer.len() {
                self.buffer[i] = self.tensor_to_color(value);
            }
        }
        
        // Update window
        self.window.update_with_buffer(&self.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .expect("Failed to update window");
    }
    
    fn handle_input(&mut self) {
        if self.window.is_key_down(Key::Escape) {
            std::process::exit(0);
        }
        
        // Visualization mode controls
        if self.window.is_key_pressed(Key::Key1, minifb::KeyRepeat::No) {
            self.visualization_mode = VisualizationMode::PsychedelicWaves;
        }
        if self.window.is_key_pressed(Key::Key2, minifb::KeyRepeat::No) {
            self.visualization_mode = VisualizationMode::CosmicSpiral;
        }
        if self.window.is_key_pressed(Key::Key3, minifb::KeyRepeat::No) {
            self.visualization_mode = VisualizationMode::TensorStorm;
        }
        if self.window.is_key_pressed(Key::Key4, minifb::KeyRepeat::No) {
            self.visualization_mode = VisualizationMode::PlasmaField;
        }
        if self.window.is_key_pressed(Key::Key5, minifb::KeyRepeat::No) {
            self.visualization_mode = VisualizationMode::QuantumRipples;
        }
        if self.window.is_key_pressed(Key::Key6, minifb::KeyRepeat::No) {
            self.visualization_mode = VisualizationMode::HypnoticMandalas;
        }
        
        // Operation controls
        if self.window.is_key_pressed(Key::S, minifb::KeyRepeat::No) {
            self.current_operation = TensorOperation::Sum;
        }
        if self.window.is_key_pressed(Key::M, minifb::KeyRepeat::No) {
            self.current_operation = TensorOperation::Max;
        }
        if self.window.is_key_pressed(Key::N, minifb::KeyRepeat::No) {
            self.current_operation = TensorOperation::Min;
        }
        if self.window.is_key_pressed(Key::C, minifb::KeyRepeat::No) {
            self.current_operation = TensorOperation::Cumsum;
        }
        
        // Backend toggle
        if self.window.is_key_pressed(Key::G, minifb::KeyRepeat::No) {
            self.use_gpu = !self.use_gpu;
        }
        
        // Auto-cycle toggles
        if self.window.is_key_pressed(Key::A, minifb::KeyRepeat::No) {
            self.auto_cycle_operations = !self.auto_cycle_operations;
        }
        if self.window.is_key_pressed(Key::V, minifb::KeyRepeat::No) {
            self.auto_cycle_modes = !self.auto_cycle_modes;
        }
        
        // Animation speed controls
        if self.window.is_key_down(Key::Up) {
            self.animation_speed = (self.animation_speed * 1.05).min(5.0);
        }
        if self.window.is_key_down(Key::Down) {
            self.animation_speed = (self.animation_speed * 0.95).max(0.1);
        }
        
        // Visual effect controls
        if self.window.is_key_down(Key::Left) {
            self.intensity_multiplier = (self.intensity_multiplier * 0.98).max(0.1);
        }
        if self.window.is_key_down(Key::Right) {
            self.intensity_multiplier = (self.intensity_multiplier * 1.02).min(3.0);
        }
        
        if self.window.is_key_down(Key::PageUp) {
            self.wave_complexity = (self.wave_complexity * 1.02).min(3.0);
        }
        if self.window.is_key_down(Key::PageDown) {
            self.wave_complexity = (self.wave_complexity * 0.98).max(0.1);
        }
        
        // Print current status
        if self.window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) {
            self.print_status();
        }
    }
    
    fn print_status(&self) {
        println!("\n🌀 Psychedelic Tensor Visualizer Status 🌀");
        println!("Visualization Mode: {}", self.visualization_mode.name());
        println!("Current Operation: {}", self.current_operation.name());
        println!("Backend: {}", if self.use_gpu { "GPU (WGPU)" } else { "CPU" });
        println!("Animation Speed: {:.2}x", self.animation_speed);
        println!("Intensity: {:.2}", self.intensity_multiplier);
        println!("Wave Complexity: {:.2}", self.wave_complexity);
        println!("Auto-cycle Operations: {}", self.auto_cycle_operations);
        println!("Auto-cycle Modes: {}", self.auto_cycle_modes);
        
        if let Ok(stats) = self.performance_stats.try_lock() {
            if let Some(avg) = stats.average_time() {
                println!("Average Operation Time: {:.2}ms", avg.as_secs_f64() * 1000.0);
            }
        }
        
        println!("\nControls:");
        println!("1-6: Switch visualization modes");
        println!("S/M/N/C: Switch operations (Sum/Max/miN/Cumsum)");
        println!("G: Toggle GPU/CPU backend");
        println!("A: Toggle auto-cycle operations");
        println!("V: Toggle auto-cycle visualization modes");
        println!("↑↓: Animation speed, ←→: Intensity, PgUp/PgDn: Wave complexity");
        println!("Space: Print status, Esc: Exit");
    }

    fn main() {
    println!("🔥🌀 Starting Burn Psychedelic Tensor Visualizer 🌀🔥");
    println!("Initializing GPU backend and psychedelic patterns...");
    
    let mut visualizer = PsychedelicTensorVisualizer::new();
    
    println!("🎨 Psychedelic Tensor Visualizer Ready! 🎨");
    println!("Press SPACE for controls, ESC to exit");
    
    visualizer.print_status();
    
    // Main render loop
    while visualizer.window.is_open() && !visualizer.window.is_key_down(Key::Escape) {
        visualizer.handle_input();
        visualizer.render();
    }
    
    println!("✨ Thanks for exploring the psychedelic tensor dimension! ✨");
}

    fn perform_gpu_operations(&self) -> (Vec<f32>, Vec<f32>, Duration) {
        let tensor = Tensor::<GpuBackend, 2>::from_data(
            TensorData::new(self.current_tensor.clone(), Shape::new([TENSOR_SIZE, TENSOR_SIZE])),
            &self.gpu_device
        );

        let start_time = Instant::now();
        
        // Perform scan operation
        let scan_config = ScanConfig::new(self.current_operation, 0);
        let scan_result = tensor.clone().scan(scan_config);
        let scan_data = scan_result.to_data().to_vec::<f32>().unwrap();
        
        // Perform FFT operation
        let fft_result = tensor.fft(0);
        let fft_data = fft_result.to_data().to_vec::<f32>().unwrap();
        
        // Take magnitude of complex FFT result
        let fft_magnitude: Vec<f32> = fft_data.chunks(2)
            .map(|chunk| (chunk[0].powi(2) + chunk[1].powi(2)).sqrt())
            .collect();
        
        let duration = start_time.elapsed();
        (scan_data, fft_magnitude, duration)
    }

    fn update_display(&mut self) {
        // Clear buffer
        self.buffer.fill(0x000000);

        // Generate fresh test pattern with some animation
        let time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f32() * 0.5;
        
        self.current_tensor = (0..TENSOR_SIZE*TENSOR_SIZE).map(|i| {
            let x = (i % TENSOR_SIZE) as f32 / TENSOR_SIZE as f32;
            let y = (i / TENSOR_SIZE) as f32 / TENSOR_SIZE as f32;
            
            let wave1 = (x * 8.0 * std::f32::consts::PI + time).sin();
            let wave2 = (y * 6.0 * std::f32::consts::PI + time * 0.7).sin();
            let ripple = ((x - 0.5).powi(2) + (y - 0.5).powi(2)).sqrt() * 10.0;
            
            (wave1 * wave2 + (ripple + time * 2.0).sin()) * 0.5 + 0.5
        }).collect();

        // Perform operations on current backend
        let (scan_data, fft_data, operation_time) = if self.use_gpu {
            self.perform_gpu_operations()
        } else {
            self.perform_cpu_operations()
        };

        // Update performance stats
        if let Ok(mut stats) = self.performance_stats.lock() {
            let throughput = (TENSOR_SIZE * TENSOR_SIZE) as f64 / operation_time.as_secs_f64() / 1_000_000.0;
            if self.use_gpu {
                stats.gpu_time = operation_time;
                stats.gpu_throughput = throughput;
            } else {
                stats.cpu_time = operation_time;
                stats.cpu_throughput = throughput;
            }
        }

        // Create visualizations
        let original_image = self.tensor_to_image(&self.current_tensor, TENSOR_SIZE, TENSOR_SIZE);
        let scan_image = self.tensor_to_image(&scan_data, TENSOR_SIZE, TENSOR_SIZE);
        let fft_image = self.tensor_to_image(&fft_data, TENSOR_SIZE, TENSOR_SIZE);

        // Composite into main buffer
        self.blit_image(&original_image, 10, 10, TENSOR_SIZE, TENSOR_SIZE);
        self.blit_image(&scan_image, TENSOR_SIZE + 20, 10, TENSOR_SIZE, TENSOR_SIZE);
        self.blit_image(&fft_image, TENSOR_SIZE * 2 + 30, 10, TENSOR_SIZE, TENSOR_SIZE);

        // Add labels and stats
        self.draw_ui();
    }

    fn blit_image(&mut self, image: &[u32], x_offset: usize, y_offset: usize, width: usize, height: usize) {
        for y in 0..height {
            for x in 0..width {
                let src_idx = y * width + x;
                let dst_idx = (y + y_offset) * WINDOW_WIDTH + (x + x_offset);
                
                if src_idx < image.len() && dst_idx < self.buffer.len() {
                    self.buffer[dst_idx] = image[src_idx];
                }
            }
        }
    }

    fn draw_ui(&mut self) {
        // Simple text rendering for stats (this would be better with a proper text library)
        let y_pos = TENSOR_SIZE + 20;
        
        // Draw backend indicator
        let backend_color = if self.use_gpu { 0x00FF00 } else { 0x0080FF };
        self.draw_text_simple(&format!("Backend: {}", if self.use_gpu { "GPU" } else { "CPU" }), 
                             10, y_pos, backend_color);
        
        // Draw operation type
        let op_name = match self.current_operation {
            ScanOp::Add => "CUMSUM",
            ScanOp::Mul => "CUMPROD", 
            ScanOp::Max => "CUMMAX",
            ScanOp::Min => "CUMMIN",
        };
        self.draw_text_simple(&format!("Operation: {}", op_name), 10, y_pos + 15, 0xFFFFFF);
        
        // Draw performance stats
        if let Ok(stats) = self.performance_stats.lock() {
            self.draw_text_simple(&format!("CPU: {:.2}ms ({:.1} Melems/s)", 
                                 stats.cpu_time.as_secs_f64() * 1000.0, stats.cpu_throughput), 
                                 10, y_pos + 30, 0x0080FF);
            self.draw_text_simple(&format!("GPU: {:.2}ms ({:.1} Melems/s)", 
                                 stats.gpu_time.as_secs_f64() * 1000.0, stats.gpu_throughput), 
                                 10, y_pos + 45, 0x00FF00);
        }
        
        // Draw labels for panels
        self.draw_text_simple("Original", 10, 0, 0xFFFFFF);
        self.draw_text_simple("Scan Result", TENSOR_SIZE + 20, 0, 0xFFFFFF);
        self.draw_text_simple("FFT Magnitude", TENSOR_SIZE * 2 + 30, 0, 0xFFFFFF);
        
        // Draw controls
        self.draw_text_simple("Controls: SPACE=Backend, 1-4=Operations, Q=Quit", 10, y_pos + 70, 0x808080);
    }

    fn draw_text_simple(&mut self, text: &str, x: usize, y: usize, color: u32) {
        // Very simple 8x8 pixel font rendering
        for (i, ch) in text.chars().enumerate() {
            self.draw_char_8x8(ch, x + i * 8, y, color);
        }
    }

    fn draw_char_8x8(&mut self, ch: char, x: usize, y: usize, color: u32) {
        // Simplified 8x8 font for basic characters
        let pattern = match ch {
            'A' => [0x18, 0x3C, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x00],
            'B' => [0x7C, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x7C, 0x00],
            'C' => [0x3C, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3C, 0x00],
            'G' => [0x3C, 0x66, 0x60, 0x6E, 0x66, 0x66, 0x3C, 0x00],
            'P' => [0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x00],
            'U' => [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00],
            'S' => [0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0x00],
            ' ' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ':' => [0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00],
            '(' => [0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00],
            ')' => [0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00],
            '0'..='9' => {
                let digit = ch as u8 - b'0';
                match digit {
                    0 => [0x3C, 0x66, 0x6E, 0x76, 0x66, 0x66, 0x3C, 0x00],
                    1 => [0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00],
                    2 => [0x3C, 0x66, 0x06, 0x1C, 0x30, 0x60, 0x7E, 0x00],
                    _ => [0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00],
                }
            },
            _ => [0xFF, 0x81, 0x81, 0x81, 0x81, 0x81, 0xFF, 0x00], // Unknown char
        };

        for (row, &byte) in pattern.iter().enumerate() {
            for col in 0..8 {
                if byte & (0x80 >> col) != 0 {
                    let px = x + col;
                    let py = y + row;
                    let idx = py * WINDOW_WIDTH + px;
                    if idx < self.buffer.len() {
                        self.buffer[idx] = color;
                    }
                }
            }
        }
    }

    fn handle_input(&mut self) {
        if self.window.is_key_down(Key::Space) {
            // Toggle backend
            self.use_gpu = !self.use_gpu;
            println!("Switched to {} backend", if self.use_gpu { "GPU" } else { "CPU" });
            thread::sleep(Duration::from_millis(200)); // Debounce
        }

        if self.window.is_key_down(Key::Key1) {
            self.current_operation = ScanOp::Add;
            println!("Operation: CUMSUM");
        }
        if self.window.is_key_down(Key::Key2) {
            self.current_operation = ScanOp::Mul;
            println!("Operation: CUMPROD");
        }
        if self.window.is_key_down(Key::Key3) {
            self.current_operation = ScanOp::Max;
            println!("Operation: CUMMAX");
        }
        if self.window.is_key_down(Key::Key4) {
            self.current_operation = ScanOp::Min;
            println!("Operation: CUMMIN");
        }
    }

    fn run(&mut self) {
        println!("🎨 Real-time Tensor Visualization Started!");
        println!("Controls:");
        println!("  SPACE - Toggle CPU/GPU backend");
        println!("  1 - Cumsum operation");
        println!("  2 - Cumprod operation");
        println!("  3 - Cummax operation");
        println!("  4 - Cummin operation");
        println!("  Q - Quit");

        while self.window.is_open() && !self.window.is_key_down(Key::Q) {
            self.handle_input();
            self.update_display();
            
            self.window
                .update_with_buffer(&self.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
                .unwrap();
                
            thread::sleep(Duration::from_millis(50)); // ~20 FPS
        }
        
        println!("Visualization ended.");
    }
}

fn main() {
    println!("🚀 Starting Real-time Tensor Visualization...");
    
    let mut visualizer = TensorVisualizer::new();
    visualizer.run();
}
