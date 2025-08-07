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

use minifb::{Key, Window, WindowOptions};
use std::time::{Duration, Instant};

// Simplified without actual tensor operations for now - focusing on pure 2D visualization

// Window constants for multi-tensor grid
const WINDOW_WIDTH: usize = 1200;
const WINDOW_HEIGHT: usize = 800;
const GRID_COLS: usize = 3;
const GRID_ROWS: usize = 2;
const CELL_WIDTH: usize = WINDOW_WIDTH / GRID_COLS;
const CELL_HEIGHT: usize = WINDOW_HEIGHT / GRID_ROWS;
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
    current_operation: TensorOperation,
    animation_speed: f32,
    intensity_multiplier: f32,
    color_phase: f32,
    last_frame_time: Instant,
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
        
        Self {
            window,
            buffer: vec![0; WINDOW_WIDTH * WINDOW_HEIGHT],
            time: 0.0,
            current_operation: TensorOperation::Sum,
            animation_speed: 1.0,
            intensity_multiplier: 1.0,
            color_phase: 0.0,
            last_frame_time: Instant::now(),
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
        // Operation controls
        if self.window.is_key_pressed(Key::S, minifb::KeyRepeat::No) {
            self.current_operation = TensorOperation::Sum;
            println!("🔄 Operation: {}", self.current_operation.name());
        }
        if self.window.is_key_pressed(Key::M, minifb::KeyRepeat::No) {
            self.current_operation = TensorOperation::Max;
            println!("🔄 Operation: {}", self.current_operation.name());
        }
        if self.window.is_key_pressed(Key::N, minifb::KeyRepeat::No) {
            self.current_operation = TensorOperation::Min;
            println!("🔄 Operation: {}", self.current_operation.name());
        }
        if self.window.is_key_pressed(Key::E, minifb::KeyRepeat::No) {
            self.current_operation = TensorOperation::Mean;
            println!("🔄 Operation: {}", self.current_operation.name());
        }
        
        // Backend toggle
        // (Removed for simplified version - pure math visualization)
        
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
            println!("Status: Op={}, Speed={:.2}, Intensity={:.2}", 
                    self.current_operation.name(),
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
        
        // Generate all 6 visualization modes simultaneously in a grid
        let modes = [
            VisualizationMode::PsychedelicWaves,
            VisualizationMode::CosmicSpiral, 
            VisualizationMode::TensorStorm,
            VisualizationMode::PlasmaField,
            VisualizationMode::QuantumRipples,
            VisualizationMode::HypnoticMandalas,
        ];
        
        for (mode_idx, &mode) in modes.iter().enumerate() {
            let grid_x = mode_idx % GRID_COLS;
            let grid_y = mode_idx / GRID_COLS;
            
            // Generate 2D tensor for this mode
            let tensor = self.generate_2d_tensor(mode);
            
            // Render this tensor into the appropriate grid cell
            self.render_tensor_to_cell(&tensor, grid_x, grid_y);
        }
        
        // Update window
        self.window.update_with_buffer(&self.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .expect("Failed to update window");
    }
    
    fn generate_2d_tensor(&self, mode: VisualizationMode) -> Vec<Vec<f32>> {
        let mut tensor = vec![vec![0.0; TENSOR_SIZE]; TENSOR_SIZE];
        
        match mode {
            VisualizationMode::PsychedelicWaves => {
                // 2D Neural oscillations across cortex
                for y in 0..TENSOR_SIZE {
                    for x in 0..TENSOR_SIZE {
                        let x_norm = x as f32 / TENSOR_SIZE as f32;
                        let y_norm = y as f32 / TENSOR_SIZE as f32;
                        let wave1 = (x_norm * 10.0 + y_norm * 8.0 + self.time * 3.0).sin();
                        let wave2 = (x_norm * 15.0 - y_norm * 12.0 - self.time * 2.0).cos();
                        let wave3 = ((x_norm - 0.5).powi(2) + (y_norm - 0.5).powi(2)).sqrt() * 20.0 + self.time * 1.5;
                        tensor[y][x] = wave1 + wave2 * 0.7 + wave3.sin() * 0.4;
                    }
                }
            }
            
            VisualizationMode::CosmicSpiral => {
                // Spiral patterns - like galaxy arms or neural pathways
                for y in 0..TENSOR_SIZE {
                    for x in 0..TENSOR_SIZE {
                        let x_center = x as f32 / TENSOR_SIZE as f32 - 0.5;
                        let y_center = y as f32 / TENSOR_SIZE as f32 - 0.5;
                        let radius = (x_center.powi(2) + y_center.powi(2)).sqrt();
                        let angle = y_center.atan2(x_center);
                        let spiral = (angle * 3.0 + radius * 15.0 + self.time * 2.0).sin();
                        tensor[y][x] = spiral * (1.0 - radius).max(0.0);
                    }
                }
            }
            
            VisualizationMode::TensorStorm => {
                // Chaotic patterns - neural storms
                for y in 0..TENSOR_SIZE {
                    for x in 0..TENSOR_SIZE {
                        let x_norm = x as f32 / TENSOR_SIZE as f32;
                        let y_norm = y as f32 / TENSOR_SIZE as f32;
                        let chaos1 = (x_norm * 25.0 + y_norm * 30.0 + self.time * 7.0).sin();
                        let chaos2 = (x_norm * 33.0 - y_norm * 28.0 - self.time * 5.0).cos();
                        let chaos3 = ((x_norm + y_norm) * 40.0 + self.time * 3.0).sin();
                        let modulation = (self.time * 1.5).cos();
                        tensor[y][x] = (chaos1 * chaos2 + chaos3) * (1.0 + modulation * 0.8);
                    }
                }
            }
            
            VisualizationMode::PlasmaField => {
                // Electromagnetic field patterns
                for y in 0..TENSOR_SIZE {
                    for x in 0..TENSOR_SIZE {
                        let x_norm = x as f32 / TENSOR_SIZE as f32;
                        let y_norm = y as f32 / TENSOR_SIZE as f32;
                        let plasma1 = (x_norm * 20.0 + self.time * 4.0).sin();
                        let plasma2 = (y_norm * 18.0 - self.time * 3.0).cos();
                        let interference = ((x_norm + y_norm) * 30.0 + self.time * 6.0).sin() * 0.5;
                        tensor[y][x] = plasma1 + plasma2 + interference;
                    }
                }
            }
            
            VisualizationMode::QuantumRipples => {
                // Quantum field fluctuations
                for y in 0..TENSOR_SIZE {
                    for x in 0..TENSOR_SIZE {
                        let x_center = x as f32 / TENSOR_SIZE as f32 - 0.5;
                        let y_center = y as f32 / TENSOR_SIZE as f32 - 0.5;
                        let distance = (x_center.powi(2) + y_center.powi(2)).sqrt();
                        let ripple = (distance * 25.0 - self.time * 5.0).sin();
                        let decay = (-distance * 6.0).exp();
                        tensor[y][x] = ripple * decay;
                    }
                }
            }
            
            VisualizationMode::HypnoticMandalas => {
                // Sacred geometry meets neural nets
                for y in 0..TENSOR_SIZE {
                    for x in 0..TENSOR_SIZE {
                        let x_center = x as f32 / TENSOR_SIZE as f32 - 0.5;
                        let y_center = y as f32 / TENSOR_SIZE as f32 - 0.5;
                        let angle = y_center.atan2(x_center);
                        let radius = (x_center.powi(2) + y_center.powi(2)).sqrt();
                        let mandala1 = (angle * 8.0 + self.time * 2.0).sin();
                        let mandala2 = (angle * 3.0 - self.time * 1.5).cos();
                        let mandala3 = (radius * 20.0 + self.time * 3.0).sin();
                        tensor[y][x] = (mandala1 + mandala2 * 0.7 + mandala3 * 0.4) * (1.0 - radius * 0.5).max(0.0);
                    }
                }
            }
        }
        
        tensor
    }
    
    fn render_tensor_to_cell(&mut self, tensor: &[Vec<f32>], grid_x: usize, grid_y: usize) {
        let start_x = grid_x * CELL_WIDTH;
        let start_y = grid_y * CELL_HEIGHT;
        
        // Scale factors to fit tensor into grid cell
        let scale_x = CELL_WIDTH as f32 / TENSOR_SIZE as f32;
        let scale_y = CELL_HEIGHT as f32 / TENSOR_SIZE as f32;
        
        for y in 0..CELL_HEIGHT {
            for x in 0..CELL_WIDTH {
                let screen_x = start_x + x;
                let screen_y = start_y + y;
                
                if screen_x < WINDOW_WIDTH && screen_y < WINDOW_HEIGHT {
                    // Map cell coordinates to tensor coordinates
                    let tensor_x = ((x as f32 / scale_x) as usize).min(TENSOR_SIZE - 1);
                    let tensor_y = ((y as f32 / scale_y) as usize).min(TENSOR_SIZE - 1);
                    
                    let mut value = tensor[tensor_y][tensor_x];
                    
                    // Apply tensor operation for additional effects
                    value = match self.current_operation {
                        TensorOperation::Sum => value * 1.5, // Amplify
                        TensorOperation::Max => value.max(0.0) * 2.0, // Positive amplify
                        TensorOperation::Min => value.min(0.0) * 2.0, // Negative amplify  
                        TensorOperation::Mean => value * 0.8, // Smooth
                    };
                    
                    // Add neural wave effects
                    let wave_x = ((screen_x as f32) * 0.02 + self.time * 3.0).sin() * 0.3;
                    let wave_y = ((screen_y as f32) * 0.03 - self.time * 2.0).cos() * 0.2;
                    let interference = ((screen_x as f32 + screen_y as f32) * 0.01 + self.time * 1.5).sin() * 0.15;
                    
                    let modulated_value = value + wave_x + wave_y + interference;
                    let color = self.tensor_to_color(modulated_value);
                    
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
