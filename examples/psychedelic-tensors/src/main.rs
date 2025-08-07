//! Psychedelic 2D Multi-Tensor Visualization 🌈✨
//! 
//! This creates a grid of 6 different 2D tensor visualizations running simultaneously,
//! representing different neural modulation patterns and embodied cognition dynamics.
//! 
//! Controls:
//! - S/M/N/E: Switch operations (Sum/Max/miN/mEan)
//! - G: Toggle GPU/CPU backend
//! - ↑↓: Animation speed, ←→: Intensity
//! - Space: Print status, Esc: Exit

use burn::backend::ndarray::NdArray;
use burn::tensor::Tensor;
use burn::tensor::TensorData;
use minifb::{Key, Window, WindowOptions};
use std::time::{Duration, Instant};

// Window constants for multi-tensor grid
const WINDOW_WIDTH: usize = 1200;
const WINDOW_HEIGHT: usize = 800;
const GRID_COLS: usize = 3;
const GRID_ROWS: usize = 2;
const CELL_MARGIN: usize = 8;
const CELL_WIDTH: usize = (WINDOW_WIDTH - (GRID_COLS + 1) * CELL_MARGIN) / GRID_COLS;
const CELL_HEIGHT: usize = (WINDOW_HEIGHT - (GRID_ROWS + 1) * CELL_MARGIN) / GRID_ROWS;
const TENSOR_SIZE: usize = 64; // 64x64 tensors for each visualization

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
            VisualizationMode::PsychedelicWaves => "Neural Waves",
            VisualizationMode::CosmicSpiral => "Cosmic Spiral",
            VisualizationMode::TensorStorm => "Chaos Storm",
            VisualizationMode::PlasmaField => "Plasma Field", 
            VisualizationMode::QuantumRipples => "Quantum Ripples",
            VisualizationMode::HypnoticMandalas => "Sacred Geometry",
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

struct PsychedelicVisualizer {
    window: Window,
    buffer: Vec<u32>,
    time: f32,
    animation_speed: f32,
    intensity_multiplier: f32,
    color_phase: f32,
    last_frame_time: Instant,
    tensors: Vec<burn::tensor::Tensor<burn::backend::wgpu::Wgpu<f32>, 2>>,
}

impl PsychedelicVisualizer {
    fn new() -> Self {
        let mut window = Window::new(
            "🌈 Psychedelic 2D Multi-Tensor Visualization - 500 Neuromodulators 🧠",
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            WindowOptions::default(),
        ).expect("Failed to create window");
        window.limit_update_rate(Some(Duration::from_millis(16))); // ~60 FPS

        // Use wgpu backend for all tensors
        use burn::backend::wgpu::Wgpu;
        use burn::tensor::Distribution;
        let device = burn::backend::wgpu::WgpuDevice::default();
        let mut tensors = Vec::new();
        let mut t = burn::tensor::Tensor::<Wgpu<f32>, 2>::random([TENSOR_SIZE, TENSOR_SIZE], Distribution::Default, &device);
        for _ in 0..6 {
            tensors.push(t.clone());
        }

        Self {
            window,
            buffer: vec![0; WINDOW_WIDTH * WINDOW_HEIGHT],
            time: 0.0,
            animation_speed: 1.0,
            intensity_multiplier: 1.0,
            color_phase: 0.0,
            last_frame_time: Instant::now(),
            tensors,
        }
    }
    
    fn run(&mut self) {
        println!("🌈 Starting Psychedelic 2D Multi-Tensor Visualization");
        println!("Controls: S/M/N/E=Operations, ↑↓=Speed, ←→=Intensity, Space=Status, Esc=Exit");
        
        while self.window.is_open() && !self.window.is_key_down(Key::Escape) {
            self.handle_input();
            self.update_animation();
            self.render_multi_tensors();
        }
    }
    
    fn handle_input(&mut self) {
    // Animation controls
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
        
    // Status
    if self.window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) {
        println!("Status: Speed={:.2}, Intensity={:.2}", 
            self.animation_speed,
            self.intensity_multiplier);
    }
    }
    
    fn update_animation(&mut self) {
        let now = Instant::now();
        let delta = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;
        
        self.time += delta * self.animation_speed;
        self.color_phase += delta * self.animation_speed * 2.0;
    }
    
    fn render_multi_tensors(&mut self) {
        // Clear buffer
        self.buffer.fill(0);

        // Pipeline: each tensor is the output of the previous, last feeds into first
        // Compute new tensors with different ops and feedback loop
        use burn::tensor::Tensor;
        use burn::tensor::backend::Backend;
        use burn::backend::wgpu::Wgpu;
        let device = burn::backend::wgpu::WgpuDevice::default();
        let mut new_tensors = Vec::with_capacity(6);
        for i in 0..6 {
            let input = if i == 0 {
                self.tensors[5].clone()
            } else {
                self.tensors[i - 1].clone()
            };
            let t = match i {
                0 => input.abs(),
                1 => input * 0.8 + 0.2,
                2 => input.clamp_min(0.2).clamp_max(0.8),
                3 => input * -1.0 + 1.0, // invert
                4 => input + Tensor::<Wgpu<f32>, 2>::random([TENSOR_SIZE, TENSOR_SIZE], burn::tensor::Distribution::Default, &device) * 0.1,
                5 => input * (self.time.sin() as f32 + 1.1),
                _ => input,
            };
            new_tensors.push(t);
        }
        self.tensors = new_tensors;

        // Collect tensor data first to avoid borrow checker issues
        let tensor_datas: Vec<Vec<f32>> = self.tensors.iter()
            .map(|t| t.to_data().as_slice().expect("to_data failed").to_vec())
            .collect();
        // Render using only the buffer and collected data
        for i in 0..6 {
            let grid_x = i % GRID_COLS;
            let grid_y = i / GRID_COLS;
            self.render_tensor_data_to_cell(&tensor_datas[i], grid_x, grid_y);
        }

        // Update window
        self.window.update_with_buffer(&self.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .expect("Failed to update window");
    }
    
    fn generate_2d_tensor(&self, mode: VisualizationMode) -> Tensor<NdArray<f32>, 2> {
        // Not used in pipeline mode
        panic!("generate_2d_tensor is not used in pipeline mode");
    }
    fn render_tensor_data_to_cell(&mut self, tensor_data: &Vec<f32>, grid_x: usize, grid_y: usize) {
        let start_x = grid_x * (CELL_WIDTH + CELL_MARGIN) + CELL_MARGIN;
        let start_y = grid_y * (CELL_HEIGHT + CELL_MARGIN) + CELL_MARGIN;
        let scale_x = CELL_WIDTH as f32 / TENSOR_SIZE as f32;
        let scale_y = CELL_HEIGHT as f32 / TENSOR_SIZE as f32;
        for y in 0..CELL_HEIGHT {
            for x in 0..CELL_WIDTH {
                let screen_x = start_x + x;
                let screen_y = start_y + y;
                if screen_x < WINDOW_WIDTH && screen_y < WINDOW_HEIGHT {
                    let tensor_x = ((x as f32 / scale_x) as usize).min(TENSOR_SIZE - 1);
                    let tensor_y = ((y as f32 / scale_y) as usize).min(TENSOR_SIZE - 1);
                    let idx = tensor_y * TENSOR_SIZE + tensor_x;
                    let value = tensor_data[idx];
                    let color = self.tensor_to_color(value);
                    self.buffer[screen_y * WINDOW_WIDTH + screen_x] = color;
                }
            }
        }
    }
    
    fn tensor_to_color(&self, value: f32) -> u32 {
        // 500 neuromodulators creating complex orbital dynamics
        let normalized = (value.tanh() + 1.0) * 0.5;
        
        // Multiple phase shifts - different neuromodulators
        let phase_r = self.color_phase;
        let phase_g = self.color_phase + 2.0943; // 120 degrees
        let phase_b = self.color_phase + 4.1888; // 240 degrees
        
        // Chaos modulation - near chaos dynamics
        let chaos_factor = (self.time * 7.0).sin() * (self.time * 11.0).cos() * 0.2;
        
        // Orbital dynamics - like planets around the sun
        let orbital_r = ((normalized * 6.28 + phase_r + chaos_factor).sin() * 0.5 + 0.5) * 255.0;
        let orbital_g = ((normalized * 6.28 + phase_g - chaos_factor).sin() * 0.5 + 0.5) * 255.0;
        let orbital_b = ((normalized * 6.28 + phase_b + chaos_factor * 0.5).sin() * 0.5 + 0.5) * 255.0;
        
        // Damping modulation - energy dissipation in living systems
        let damping = 1.0 + (self.time * 5.0).sin() * 0.3;
        let energy_mod = self.intensity_multiplier * damping;
        
        let r = (orbital_r * energy_mod).min(255.0) as u32;
        let g = (orbital_g * energy_mod).min(255.0) as u32;
        let b = (orbital_b * energy_mod).min(255.0) as u32;
        
        (255 << 24) | (r << 16) | (g << 8) | b
    }
}

fn main() {
    let mut visualizer = PsychedelicVisualizer::new();
    visualizer.run();
}
