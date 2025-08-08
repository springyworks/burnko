//! Realtime Tensor Visualizer with a 6-tensor closed feedback grid.

mod viz {
    use burn::{
        tensor::{Tensor, backend::Backend},
        backend::{ndarray::{NdArray, NdArrayDevice}, wgpu::{Wgpu, WgpuDevice}},
    };
    use minifb::{Key, Window, WindowOptions, Scale};
    use std::time::Duration;
    use std::sync::{Arc, Mutex};

    // Window and visualization constants
    pub const WINDOW_WIDTH: usize = 1200;
    pub const WINDOW_HEIGHT: usize = 800;

    // Grid constants for 6-tensor view
    const GRID_COLS: usize = 3;
    const GRID_ROWS: usize = 2;
    const CELL_MARGIN: usize = 8;
    const GRID_TENSOR_SIZE: usize = 64; // 64x64 tensors for grid view
    const CELL_WIDTH: usize = (WINDOW_WIDTH - (GRID_COLS + 1) * CELL_MARGIN) / GRID_COLS;
    const CELL_HEIGHT: usize = (WINDOW_HEIGHT - (GRID_ROWS + 1) * CELL_MARGIN) / GRID_ROWS;

    type CpuBackend = NdArray<f32>;
    type GpuBackend = Wgpu<f32>;
    type CpuDevice = NdArrayDevice;
    type GpuDevice = WgpuDevice;

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
    }

    #[derive(Clone)]
    struct PerformanceStats { times: Vec<Duration> }
    impl PerformanceStats { fn new() -> Self { Self { times: Vec::new() } } }

    pub struct Visualizer {
        window: Window,
        buffer: Vec<u32>,
        cpu_device: CpuDevice,
        gpu_device: GpuDevice,
        time: f32,
        animation_speed: f32,
        visualization_mode: VisualizationMode,
        use_gpu: bool,
        color_phase: f32,
        intensity_multiplier: f32,
        wave_complexity: f32,
        performance_stats: Arc<Mutex<PerformanceStats>>,
        grid_tensors_gpu: Vec<Tensor<GpuBackend, 2>>,
        grid_tensors_cpu: Vec<Tensor<CpuBackend, 2>>,
    }

    impl Visualizer {
        pub fn new() -> Self {
            let mut window = Window::new(
                "🔥 Burn 6-Tensor Pipeline Visualizer 🌀",
                WINDOW_WIDTH,
                WINDOW_HEIGHT,
                WindowOptions { resize: false, scale: Scale::X1, ..WindowOptions::default() },
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
                use_gpu: true,
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
        }

        fn tensor_to_color(&self, value: f32) -> u32 {
            let normalized = (value.tanh() + 1.0) * 0.5; // Normalize to [0, 1]
            let phase_r = self.color_phase;
            let phase_g = self.color_phase + 2.0943; // 120 degrees
            let phase_b = self.color_phase + 4.1888; // 240 degrees
            let r = ((normalized * 6.28 + phase_r).sin() * 0.5 + 0.5) * 255.0;
            let g = ((normalized * 6.28 + phase_g).sin() * 0.5 + 0.5) * 255.0;
            let b = ((normalized * 6.28 + phase_b).sin() * 0.5 + 0.5) * 255.0;
            let intensity = self.intensity_multiplier * (1.0 + (self.time * 4.0).sin() * 0.2);
            let r = (r * intensity).min(255.0) as u32;
            let g = (g * intensity).min(255.0) as u32;
            let b = (b * intensity).min(255.0) as u32;
            (255 << 24) | (r << 16) | (g << 8) | b
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

        fn handle_input(&mut self) {
            if self.window.is_key_down(Key::Escape) { std::process::exit(0); }
            // Backend toggle
            if self.window.is_key_pressed(Key::G, minifb::KeyRepeat::No) {
                self.use_gpu = !self.use_gpu;
                println!("Backend: {}", if self.use_gpu { "GPU (WGPU)" } else { "CPU (NdArray)" });
            }
            // Controls similar to previous impl
            if self.window.is_key_down(Key::Up) { self.animation_speed = (self.animation_speed * 1.05).min(5.0); }
            if self.window.is_key_down(Key::Down) { self.animation_speed = (self.animation_speed * 0.95).max(0.1); }
            if self.window.is_key_down(Key::Left) { self.intensity_multiplier = (self.intensity_multiplier * 0.98).max(0.1); }
            if self.window.is_key_down(Key::Right) { self.intensity_multiplier = (self.intensity_multiplier * 1.02).min(3.0); }
            if self.window.is_key_down(Key::PageUp) { self.wave_complexity = (self.wave_complexity * 1.02).min(3.0); }
            if self.window.is_key_down(Key::PageDown) { self.wave_complexity = (self.wave_complexity * 0.98).max(0.1); }
            if self.window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) { self.print_status(); }
        }

        fn print_status(&self) {
            println!("\n🌀 6-Tensor Pipeline Status 🌀");
            println!("Backend: {}", if self.use_gpu { "GPU (WGPU)" } else { "CPU (NdArray)" });
            println!("Animation Speed: {:.2}x", self.animation_speed);
            println!("Intensity: {:.2}", self.intensity_multiplier);
            println!("Wave Complexity: {:.2}", self.wave_complexity);
        }

        pub fn run_loop(&mut self) {
            println!("🔥🌀 Starting Burn 6-Tensor Pipeline Visualizer 🌀🔥");
            println!("Ready. Press G to switch backend, Esc to exit.");
            self.print_status();
            while self.window.is_open() && !self.window.is_key_down(Key::Escape) {
                self.handle_input();
                self.update_animation_state();
                self.render_six_tensor_grid();
            }
            println!("✨ Bye!");
        }
    }

    pub fn run() {
        let mut v = Visualizer::new();
        v.run_loop();
    }
}

pub fn run() { viz::run() }
