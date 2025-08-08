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

use burn::{
    tensor::{Tensor, backend::Backend},
    backend::{ndarray::{NdArray, NdArrayDevice}, wgpu::{Wgpu, WgpuDevice}},
};
use minifb::{Key, Window, WindowOptions, Scale};
use std::time::Duration;
use std::sync::{Arc, Mutex};

// Add: grid constants for 6-tensor view
const GRID_COLS: usize = 3;
const GRID_ROWS: usize = 2;
const CELL_MARGIN: usize = 8;
const GRID_TENSOR_SIZE: usize = 128; // was 64; larger tensors for higher detail
const CELL_WIDTH: usize = (WINDOW_WIDTH - (GRID_COLS + 1) * CELL_MARGIN) / GRID_COLS;
const CELL_HEIGHT: usize = (WINDOW_HEIGHT - (GRID_ROWS + 1) * CELL_MARGIN) / GRID_ROWS;

type CpuBackend = NdArray<f32>;
type GpuBackend = Wgpu<f32>;
type CpuDevice = NdArrayDevice;
type GpuDevice = WgpuDevice;

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
    fn name(self) -> &'static str {
        match self {
            VisualizationMode::PsychedelicWaves => "Psychedelic Waves",
            VisualizationMode::CosmicSpiral => "Cosmic Spiral",
            VisualizationMode::TensorStorm => "Tensor Storm",
            VisualizationMode::PlasmaField => "Plasma Field", 
            VisualizationMode::QuantumRipples => "Quantum Ripples",
            VisualizationMode::HypnoticMandalas => "Hypnotic Mandalas",
        }
    }
    
    fn next(self) -> Self {
        match self {
            VisualizationMode::PsychedelicWaves => VisualizationMode::CosmicSpiral,
            VisualizationMode::CosmicSpiral => VisualizationMode::TensorStorm,
            VisualizationMode::TensorStorm => VisualizationMode::PlasmaField,
            VisualizationMode::PlasmaField => VisualizationMode::QuantumRipples,
            VisualizationMode::QuantumRipples => VisualizationMode::HypnoticMandalas,
            VisualizationMode::HypnoticMandalas => VisualizationMode::PsychedelicWaves,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum TensorOperation {
    Sum,
    Max,
    Min,
    Mean,
}

impl TensorOperation {
    fn name(self) -> &'static str {
        match self {
            TensorOperation::Sum => "Sum",
            TensorOperation::Max => "Max",
            TensorOperation::Min => "Min",
            TensorOperation::Mean => "Mean",
        }
    }
    
    fn next(self) -> Self {
        match self {
            TensorOperation::Sum => TensorOperation::Max,
            TensorOperation::Max => TensorOperation::Min,
            TensorOperation::Min => TensorOperation::Mean,
            TensorOperation::Mean => TensorOperation::Sum,
        }
    }
}

// Add: view mode to merge both examples
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViewMode {
    SingleCanvas,
    SixTensorGrid,
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

struct PsychedelicTensorVisualizer {
    window: Window,
    buffer: Vec<u32>,
    cpu_device: CpuDevice,
    gpu_device: GpuDevice,
    
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

    // 6-tensor grid pipeline (both backends)
    grid_tensors_gpu: Vec<Tensor<GpuBackend, 2>>,
    grid_tensors_cpu: Vec<Tensor<CpuBackend, 2>>,
}

impl PsychedelicTensorVisualizer {
    fn new() -> Self {
        let mut window = Window::new(
            "🔥 Burn 6-Tensor Pipeline Visualizer 🌀",
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
        let cpu_device = CpuDevice::default();
        let gpu_device = GpuDevice::default();
        
        // Initialize 6 grid tensors on both backends
        use burn::tensor::Distribution;
        let mut grid_tensors_gpu = Vec::with_capacity(6);
        let t0_gpu: Tensor<GpuBackend, 2> = Tensor::random([GRID_TENSOR_SIZE, GRID_TENSOR_SIZE], Distribution::Default, &gpu_device);
        for _ in 0..6 { grid_tensors_gpu.push(t0_gpu.clone()); }

        let mut grid_tensors_cpu = Vec::with_capacity(6);
        let t0_cpu: Tensor<CpuBackend, 2> = Tensor::random([GRID_TENSOR_SIZE, GRID_TENSOR_SIZE], Distribution::Default, &cpu_device);
        for _ in 0..6 { grid_tensors_cpu.push(t0_cpu.clone()); }
        
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
            grid_tensors_gpu,
            grid_tensors_cpu,
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
        let tensor: Tensor<GpuBackend, 1> = 
            Tensor::<GpuBackend, 1>::from_floats(data.as_slice(), &self.gpu_device)
                .reshape([TENSOR_HEIGHT * TENSOR_WIDTH]);
        
        // Reshape to 2D for operations
        let tensor_2d = tensor.reshape([TENSOR_HEIGHT, TENSOR_WIDTH]);
        
        let result_tensor = match self.current_operation {
            TensorOperation::Sum => tensor_2d.sum_dim(1),
            TensorOperation::Max => tensor_2d.max_dim(1),
            TensorOperation::Min => tensor_2d.min_dim(1),
            TensorOperation::Mean => tensor_2d.mean_dim(1),
        };
        
        // Convert back to full 2D for visualization
        let result_data: Vec<f32> = result_tensor.to_data().to_vec().unwrap();
        self.expand_to_full_size(result_data)
    }
    
    fn apply_cpu_operation(&self, data: Vec<f32>) -> Vec<f32> {
        let tensor: Tensor<CpuBackend, 1> = 
            Tensor::<CpuBackend, 1>::from_floats(data.as_slice(), &self.cpu_device)
                .reshape([TENSOR_HEIGHT * TENSOR_WIDTH]);
        
        // Reshape to 2D for operations
        let tensor_2d = tensor.reshape([TENSOR_HEIGHT, TENSOR_WIDTH]);
        
        let result_tensor = match self.current_operation {
            TensorOperation::Sum => tensor_2d.sum_dim(1),
            TensorOperation::Max => tensor_2d.max_dim(1),
            TensorOperation::Min => tensor_2d.min_dim(1),
            TensorOperation::Mean => tensor_2d.mean_dim(1),
        };
        
        let result_data: Vec<f32> = result_tensor.to_data().to_vec().unwrap();
        self.expand_to_full_size(result_data)
    }
    
    fn expand_to_full_size(&self, data: Vec<f32>) -> Vec<f32> {
        // Expand the reduced tensor back to full visualization size
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
        self.render_six_tensor_grid();
    }

    fn render_six_tensor_grid(&mut self) {
        // Clear buffer
        self.buffer.fill(0);

        if self.use_gpu {
            self.update_grid_pipeline_gpu();
            let tensor_datas: Vec<Vec<f32>> = self.grid_tensors_gpu.iter()
                .map(|t| t.to_data().as_slice().expect("to_data failed").to_vec())
                .collect();
            for i in 0..6 { self.draw_cell(&tensor_datas[i], i); }
        } else {
            self.update_grid_pipeline_cpu();
            let tensor_datas: Vec<Vec<f32>> = self.grid_tensors_cpu.iter()
                .map(|t| t.to_data().as_slice().expect("to_data failed").to_vec())
                .collect();
            for i in 0..6 { self.draw_cell(&tensor_datas[i], i); }
        }

        self.window.update_with_buffer(&self.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .expect("Failed to update window");
    }

    fn draw_cell(&mut self, tensor_data: &Vec<f32>, idx: usize) {
        let grid_x = idx % GRID_COLS;
        let grid_y = idx / GRID_COLS;
        let start_x = grid_x * (CELL_WIDTH + CELL_MARGIN) + CELL_MARGIN;
        let start_y = grid_y * (CELL_HEIGHT + CELL_MARGIN) + CELL_MARGIN;
        let scale_x = CELL_WIDTH as f32 / GRID_TENSOR_SIZE as f32;
        let scale_y = CELL_HEIGHT as f32 / GRID_TENSOR_SIZE as f32;
        for y in 0..CELL_HEIGHT {
            for x in 0..CELL_WIDTH {
                let screen_x = start_x + x;
                let screen_y = start_y + y;
                if screen_x < WINDOW_WIDTH && screen_y < WINDOW_HEIGHT {
                    let tensor_x = ((x as f32 / scale_x) as usize).min(GRID_TENSOR_SIZE - 1);
                    let tensor_y = ((y as f32 / scale_y) as usize).min(GRID_TENSOR_SIZE - 1);
                    let i = tensor_y * GRID_TENSOR_SIZE + tensor_x;
                    let value = tensor_data[i];
                    self.buffer[screen_y * WINDOW_WIDTH + screen_x] = self.tensor_to_color(value);
                }
            }
        }
    }

    fn update_grid_pipeline_gpu(&mut self) {
        use burn::tensor::Distribution;
        let device = &self.gpu_device;
        let mut new_tensors = Vec::with_capacity(6);
        for i in 0..6 {
            let input = if i == 0 { self.grid_tensors_gpu[5].clone() } else { self.grid_tensors_gpu[i - 1].clone() };
            let t = match i {
                0 => input.abs(),
                1 => input * 0.8 + 0.2,
                2 => input.clamp_min(0.2).clamp_max(0.8),
                3 => input * -1.0 + 1.0,
                4 => input + Tensor::<GpuBackend, 2>::random([GRID_TENSOR_SIZE, GRID_TENSOR_SIZE], Distribution::Default, device) * 0.1,
                5 => input * ((self.time as f32).sin() + 1.1),
                _ => input,
            };
            new_tensors.push(t);
        }
        self.grid_tensors_gpu = new_tensors;
    }

    fn update_grid_pipeline_cpu(&mut self) {
        use burn::tensor::Distribution;
        let device = &self.cpu_device;
        let mut new_tensors = Vec::with_capacity(6);
        for i in 0..6 {
            let input = if i == 0 { self.grid_tensors_cpu[5].clone() } else { self.grid_tensors_cpu[i - 1].clone() };
            let t = match i {
                0 => input.abs(),
                1 => input * 0.8 + 0.2,
                2 => input.clamp_min(0.2).clamp_max(0.8),
                3 => input * -1.0 + 1.0,
                4 => input + Tensor::<CpuBackend, 2>::random([GRID_TENSOR_SIZE, GRID_TENSOR_SIZE], Distribution::Default, device) * 0.1,
                5 => input * ((self.time as f32).sin() + 1.1),
                _ => input,
            };
            new_tensors.push(t);
        }
        self.grid_tensors_cpu = new_tensors;
    }

    fn handle_input(&mut self) {
        if self.window.is_key_down(Key::Escape) { std::process::exit(0); }
        
        // Backend toggle
        if self.window.is_key_pressed(Key::G, minifb::KeyRepeat::No) {
            self.use_gpu = !self.use_gpu;
            println!("Backend: {}", if self.use_gpu { "GPU (WGPU)" } else { "CPU (NdArray)" });
        }
        
        // Visualization adjustments
        if self.window.is_key_down(Key::Up) {
            self.animation_speed = (self.animation_speed * 1.05).min(5.0);
        }
        if self.window.is_key_down(Key::Down) {
            self.animation_speed = (self.animation_speed * 0.95).max(0.1);
        }
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
        
        if self.window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) {
            self.print_status();
        }
    }
    
    fn print_status(&self) {
        println!("\n🌀 6-Tensor Pipeline Status 🌀");
        println!("Backend: {}", if self.use_gpu { "GPU (WGPU)" } else { "CPU (NdArray)" });
        println!("Animation Speed: {:.2}x", self.animation_speed);
        println!("Intensity: {:.2}", self.intensity_multiplier);
        println!("Wave Complexity: {:.2}", self.wave_complexity);
    }
}

fn main() {
    // Ensure Rayon global pool has at least 2 threads for NdArray backend
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2).max(2);
    let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global();
    println!("🔥🌀 Starting Burn 6-Tensor Pipeline Visualizer 🌀🔥");
    println!("Initializing backends...");
    let mut visualizer = PsychedelicTensorVisualizer::new();
    println!("Ready. Press G to switch backend, Esc to exit.");
    visualizer.print_status();
    while visualizer.window.is_open() && !visualizer.window.is_key_down(Key::Escape) {
        visualizer.handle_input();
        visualizer.render();
    }
    println!("✨ Bye!");
}
