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
//! 
//! Controls:
//! - 1-6: Switch visualization modes (Waves, Spiral, Storm, Plasma, Ripples, Mandalas)
//! - S/M/N/E: Switch operations (Sum/Max/miN/mEan)
//! - G: Toggle GPU/CPU backend
//! - A: Toggle auto-cycle operations
//! - V: Toggle auto-cycle visualization modes
//! - ↑↓: Animation speed, ←→: Intensity, PgUp/PgDn: Wave complexity
//! - Space: Print status, Esc: Exit

use burn::{
    backend::{ndarray::NdArray, wgpu::Wgpu},
    tensor::Tensor,
};
use minifb::{Key, Window, WindowOptions, Scale};
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};

type CpuBackend = NdArray<f32>;
type GpuBackend = Wgpu<f32>;

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

impl PsychedelicTensorVisualizer {
    fn new() -> Self {
        let mut window = Window::new(
            "🔥 Burn Psychedelic Tensor Visualizer 🌀 - Dancing Neural Dynamics",
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
        
        Self {
            window,
            buffer,
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
    
    fn generate_psychedelic_tensor(&self, backend: &Backend) -> Tensor<Backend, 2> {
        let height = 64;
        let width = 64;
        
        match self.mode {
            VisualizationMode::PsychedelicWaves => {
                // 2D Flowing wave patterns - like neural oscillations across cortex
                let data: Vec<f32> = (0..height)
                    .flat_map(|y| {
                        (0..width).map(move |x| {
                            let x_norm = x as f32 / width as f32;
                            let y_norm = y as f32 / height as f32;
                            let wave1 = (x_norm * 10.0 + y_norm * 8.0 + self.time * 3.0).sin();
                            let wave2 = (x_norm * 15.0 - y_norm * 12.0 - self.time * 2.0).cos();
                            let wave3 = ((x_norm - 0.5).powi(2) + (y_norm - 0.5).powi(2)).sqrt() * 20.0 + self.time * 1.5;
                            wave1 + wave2 * 0.7 + wave3.sin() * 0.4
                        })
                    })
                    .collect();
                Tensor::<Backend, 2>::from_floats(data.as_slice(), backend).reshape([height, width])
            }
            
            VisualizationMode::CosmicSpiral => {
                // 2D Spiral patterns - like galaxy arms or neural pathways
                let data: Vec<f32> = (0..height)
                    .flat_map(|y| {
                        (0..width).map(move |x| {
                            let x_center = x as f32 / width as f32 - 0.5;
                            let y_center = y as f32 / height as f32 - 0.5;
                            let radius = (x_center.powi(2) + y_center.powi(2)).sqrt();
                            let angle = y_center.atan2(x_center);
                            let spiral = (angle * 3.0 + radius * 15.0 + self.time * 2.0).sin();
                            spiral * (1.0 - radius).max(0.0)
                        })
                    })
                    .collect();
                Tensor::<Backend, 2>::from_floats(data.as_slice(), backend).reshape([height, width])
            }
            
            VisualizationMode::TensorStorm => {
                // 2D Chaotic patterns - representing neural storms
                let data: Vec<f32> = (0..height)
                    .flat_map(|y| {
                        (0..width).map(move |x| {
                            let x_norm = x as f32 / width as f32;
                            let y_norm = y as f32 / height as f32;
                            let chaos1 = (x_norm * 25.0 + y_norm * 30.0 + self.time * 7.0).sin();
                            let chaos2 = (x_norm * 33.0 - y_norm * 28.0 - self.time * 5.0).cos();
                            let chaos3 = ((x_norm + y_norm) * 40.0 + self.time * 3.0).sin();
                            let modulation = (self.time * 1.5).cos();
                            (chaos1 * chaos2 + chaos3) * (1.0 + modulation * 0.8)
                        })
                    })
                    .collect();
                Tensor::<Backend, 2>::from_floats(data.as_slice(), backend).reshape([height, width])
            }
            
            VisualizationMode::PlasmaField => {
                // 2D Plasma-like patterns - electromagnetic fields
                let data: Vec<f32> = (0..height)
                    .flat_map(|y| {
                        (0..width).map(move |x| {
                            let x_norm = x as f32 / width as f32;
                            let y_norm = y as f32 / height as f32;
                            let plasma1 = (x_norm * 20.0 + self.time * 4.0).sin();
                            let plasma2 = (y_norm * 18.0 - self.time * 3.0).cos();
                            let interference = ((x_norm + y_norm) * 30.0 + self.time * 6.0).sin() * 0.5;
                            plasma1 + plasma2 + interference
                        })
                    })
                    .collect();
                Tensor::<Backend, 2>::from_floats(data.as_slice(), backend).reshape([height, width])
            }
            
            VisualizationMode::QuantumRipples => {
                // 2D Ripple patterns - like quantum field fluctuations
                let data: Vec<f32> = (0..height)
                    .flat_map(|y| {
                        (0..width).map(move |x| {
                            let x_center = x as f32 / width as f32 - 0.5;
                            let y_center = y as f32 / height as f32 - 0.5;
                            let distance = (x_center.powi(2) + y_center.powi(2)).sqrt();
                            let ripple = (distance * 25.0 - self.time * 5.0).sin();
                            let decay = (-distance * 6.0).exp();
                            ripple * decay
                        })
                    })
                    .collect();
                Tensor::<Backend, 2>::from_floats(data.as_slice(), backend).reshape([height, width])
            }
            
            VisualizationMode::HypnoticMandalas => {
                // 2D Mandala patterns - sacred geometry meets neural nets
                let data: Vec<f32> = (0..height)
                    .flat_map(|y| {
                        (0..width).map(move |x| {
                            let x_center = x as f32 / width as f32 - 0.5;
                            let y_center = y as f32 / height as f32 - 0.5;
                            let angle = y_center.atan2(x_center);
                            let radius = (x_center.powi(2) + y_center.powi(2)).sqrt();
                            let mandala1 = (angle * 8.0 + self.time * 2.0).sin();
                            let mandala2 = (angle * 3.0 - self.time * 1.5).cos();
                            let mandala3 = (radius * 20.0 + self.time * 3.0).sin();
                            (mandala1 + mandala2 * 0.7 + mandala3 * 0.4) * (1.0 - radius * 0.5).max(0.0)
                        })
                    })
                    .collect();
                Tensor::<Backend, 2>::from_floats(data.as_slice(), backend).reshape([height, width])
            }
        }
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
        let device = Default::default();
        let tensor = Tensor::<GpuBackend, 1>::from_floats(data.as_slice(), &device)
            .reshape([TENSOR_HEIGHT, TENSOR_WIDTH]);
        
        let result_tensor = match self.current_operation {
            TensorOperation::Sum => tensor.sum_dim(1),
            TensorOperation::Max => tensor.max_dim(1),
            TensorOperation::Min => tensor.min_dim(1),
            TensorOperation::Mean => tensor.mean_dim(1),
        };
        
        // Convert back to full 2D for visualization
        let result_data: Vec<f32> = result_tensor.into_data().to_vec().unwrap();
        self.expand_to_full_size(result_data)
    }
    
    fn apply_cpu_operation(&self, data: Vec<f32>) -> Vec<f32> {
        let device = Default::default();
        let tensor = Tensor::<CpuBackend, 1>::from_floats(data.as_slice(), &device)
            .reshape([TENSOR_HEIGHT, TENSOR_WIDTH]);
        
        let result_tensor = match self.current_operation {
            TensorOperation::Sum => tensor.sum_dim(1),
            TensorOperation::Max => tensor.max_dim(1),
            TensorOperation::Min => tensor.min_dim(1),
            TensorOperation::Mean => tensor.mean_dim(1),
        };
        
        let result_data: Vec<f32> = result_tensor.into_data().to_vec().unwrap();
        self.expand_to_full_size(result_data)
    }
    
    fn expand_to_full_size(&self, data: Vec<f32>) -> Vec<f32> {
        // Create more dynamic expansion - like neural activation spreading
        let mut full_data = vec![0.0; TENSOR_HEIGHT * TENSOR_WIDTH];
        
        for y in 0..TENSOR_HEIGHT {
            let base_value = if y < data.len() { data[y] } else { 0.0 };
            
            for x in 0..TENSOR_WIDTH {
                // Add wave propagation effects - like neural signals spreading
                let wave_x = (x as f32 / TENSOR_WIDTH as f32 * 8.0 + self.time * 3.0).sin();
                let wave_y = (y as f32 / TENSOR_HEIGHT as f32 * 6.0 + self.time * 2.0).cos();
                
                // Create interference patterns - like neuromodulator interactions
                let interference = (wave_x * wave_y + self.time * 4.0).sin() * 0.3;
                
                // Modulate the base value with the wave patterns
                let modulated_value = base_value * (1.0 + interference * self.wave_complexity);
                
                full_data[y * TENSOR_WIDTH + x] = modulated_value;
            }
        }
        
        full_data
    }
    
    fn tensor_to_color(&self, value: f32) -> u32 {
        // Psychedelic color mapping - 500 neuromodulators creating complex patterns
        let normalized = (value.tanh() + 1.0) * 0.5; // Normalize to [0, 1]
        
        // Multiple phase shifts - like different neuromodulators
        let phase_r = self.color_phase;
        let phase_g = self.color_phase + 2.0943; // 120 degrees
        let phase_b = self.color_phase + 4.1888; // 240 degrees
        
        // Add chaos modulation - near chaos dynamics you mentioned
        let chaos_factor = (self.time * 7.0).sin() * (self.time * 11.0).cos() * 0.2;
        
        // Create orbital dynamics - like planets around the sun
        let orbital_r = ((normalized * 6.28 + phase_r + chaos_factor).sin() * 0.5 + 0.5) * 255.0;
        let orbital_g = ((normalized * 6.28 + phase_g - chaos_factor).sin() * 0.5 + 0.5) * 255.0;
        let orbital_b = ((normalized * 6.28 + phase_b + chaos_factor * 0.5).sin() * 0.5 + 0.5) * 255.0;
        
        // Add damping modulation - like energy dissipation in living systems
        let damping = 1.0 + (self.time * 5.0).sin() * 0.3;
        let energy_mod = self.intensity_multiplier * damping;
        
        let r = (orbital_r * energy_mod).min(255.0) as u32;
        let g = (orbital_g * energy_mod).min(255.0) as u32;
        let b = (orbital_b * energy_mod).min(255.0) as u32;
        
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
        if self.window.is_key_pressed(Key::E, minifb::KeyRepeat::No) {
            self.current_operation = TensorOperation::Mean;
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
        println!("\n🌀 Psychedelic Tensor Visualizer - Dancing Neural Dynamics 🌀");
        println!("🧠 Embodied Cognition: Brain as Dancer, Energy as Movement");
        println!("⚡ Visualization Mode: {}", self.visualization_mode.name());
        println!("🔥 Current Operation: {}", self.current_operation.name());
        println!("🚀 Backend: {}", if self.use_gpu { "GPU (WGPU)" } else { "CPU" });
        println!("🌊 Animation Speed: {:.2}x", self.animation_speed);
        println!("✨ Intensity: {:.2}", self.intensity_multiplier);
        println!("🌈 Wave Complexity: {:.2}", self.wave_complexity);
        println!("🔄 Auto-cycle Operations: {}", self.auto_cycle_operations);
        println!("🎭 Auto-cycle Modes: {}", self.auto_cycle_modes);
        
        if let Ok(stats) = self.performance_stats.try_lock() {
            if let Some(avg) = stats.average_time() {
                println!("⏱️  Average Operation Time: {:.2}ms", avg.as_secs_f64() * 1000.0);
            }
        }
        
        println!("\n🎮 Controls - Dance with the Tensors:");
        println!("1-6: Visualization modes (Waves→Spiral→Storm→Plasma→Ripples→Mandalas)");
        println!("S/M/N/E: Operations (Sum/Max/miN/mEan)");
        println!("G: Toggle GPU/CPU backend");
        println!("A: Toggle auto-cycle operations");
        println!("V: Toggle auto-cycle visualization modes");
        println!("↑↓: Animation speed, ←→: Intensity, PgUp/PgDn: Wave complexity");
        println!("Space: Print status, Esc: Exit");
        println!("\n🌟 Experience the dance of 500 neuromodulators through tensor space! 🌟");
    }
}

fn main() {
    println!("🔥🌀 Starting Burn Psychedelic Tensor Visualizer 🌀🔥");
    println!("🧠 Simulating Embodied Cognition & Neural Dynamics");
    println!("⚡ Modulating Energy, Chaos, and Damping through Tensor Space");
    println!("🌟 The Brain Dances, We Dance with It!");
    println!();
    println!("Initializing GPU backend and psychedelic patterns...");
    
    let mut visualizer = PsychedelicTensorVisualizer::new();
    
    println!("🎨 Psychedelic Tensor Visualizer Ready! 🎨");
    println!("Press SPACE for controls, ESC to exit");
    println!("Let the dance of neural dynamics begin! 💃🕺");
    
    visualizer.print_status();
    
    // Main render loop - the eternal dance
    while visualizer.window.is_open() && !visualizer.window.is_key_down(Key::Escape) {
        visualizer.handle_input();
        visualizer.render();
    }
    
    println!("✨ Thanks for exploring the psychedelic tensor dimension! ✨");
    println!("🧠 Remember: We started as dancers, and dance we will! 💃🕺");
    println!("🌟 500 neuromodulators, infinite possibilities! 🌟");
}
