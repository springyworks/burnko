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

    // New: ripple modes
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum RippleMode { Analytic, Stateful }

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
        // New: ripple infra
        ripple_mode: RippleMode,
        gain_ripple: f32,
        gain_feedback: f32,
        coords_gpu_x: Tensor<GpuBackend, 2>,
        coords_gpu_y: Tensor<GpuBackend, 2>,
        coords_cpu_x: Tensor<CpuBackend, 2>,
        coords_cpu_y: Tensor<CpuBackend, 2>,
        ripple_state_gpu: Tensor<GpuBackend, 2>,
        ripple_state_cpu: Tensor<CpuBackend, 2>,
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

            // Build static coordinate fields on each backend [0,1] range
            let (coords_gpu_x, coords_gpu_y) = Self::build_coords_gpu(&gpu_device);
            let (coords_cpu_x, coords_cpu_y) = Self::build_coords_cpu(&cpu_device);

            // Ripple states (zero)
            let zero_gpu = Tensor::<GpuBackend, 2>::from_floats(&vec![0.0; GRID_TENSOR_SIZE * GRID_TENSOR_SIZE], &gpu_device)
                .reshape([GRID_TENSOR_SIZE, GRID_TENSOR_SIZE]);
            let zero_cpu = Tensor::<CpuBackend, 2>::from_floats(&vec![0.0; GRID_TENSOR_SIZE * GRID_TENSOR_SIZE], &cpu_device)
                .reshape([GRID_TENSOR_SIZE, GRID_TENSOR_SIZE]);

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
                ripple_mode: RippleMode::Analytic,
                gain_ripple: 0.7,
                gain_feedback: 0.6,
                coords_gpu_x,
                coords_gpu_y,
                coords_cpu_x,
                coords_cpu_y,
                ripple_state_gpu: zero_gpu,
                ripple_state_cpu: zero_cpu,
            }
        }

        fn build_coords_gpu(device: &GpuDevice) -> (Tensor<GpuBackend, 2>, Tensor<GpuBackend, 2>) {
            let w = GRID_TENSOR_SIZE;
            let h = GRID_TENSOR_SIZE;
            let mut xs = Vec::with_capacity(w * h);
            let mut ys = Vec::with_capacity(w * h);
            for j in 0..h {
                for i in 0..w {
                    xs.push(i as f32 / (w - 1) as f32);
                    ys.push(j as f32 / (h - 1) as f32);
                }
            }
            let x = Tensor::<GpuBackend, 2>::from_floats(&xs, device).reshape([h, w]);
            let y = Tensor::<GpuBackend, 2>::from_floats(&ys, device).reshape([h, w]);
            (x, y)
        }
        fn build_coords_cpu(device: &CpuDevice) -> (Tensor<CpuBackend, 2>, Tensor<CpuBackend, 2>) {
            let w = GRID_TENSOR_SIZE;
            let h = GRID_TENSOR_SIZE;
            let mut xs = Vec::with_capacity(w * h);
            let mut ys = Vec::with_capacity(w * h);
            for j in 0..h {
                for i in 0..w {
                    xs.push(i as f32 / (w - 1) as f32);
                    ys.push(j as f32 / (h - 1) as f32);
                }
            }
            let x = Tensor::<CpuBackend, 2>::from_floats(&xs, device).reshape([h, w]);
            let y = Tensor::<CpuBackend, 2>::from_floats(&ys, device).reshape([h, w]);
            (x, y)
        }

        // Analytic ripple on GPU
        fn ripple_analytic_gpu(&self) -> Tensor<GpuBackend, 2> {
            let cx = 0.5 + 0.25 * (self.time * 0.7).cos();
            let cy = 0.5 + 0.25 * (self.time * 0.9).sin();
            let k = 20.0;      // spatial frequency
            let omega = 6.0;   // speed
            let damping = 1.5; // radial damping
            let amp = 1.0;
            let dx = &self.coords_gpu_x - cx;
            let dy = &self.coords_gpu_y - cy;
            let r = (dx.clone() * dx + dy.clone() * dy).sqrt();
            let phase = r.clone() * k - self.time * omega;
            let env = (r * -damping).exp();
            phase.sin() * env * amp
        }
        // Analytic ripple on CPU
        fn ripple_analytic_cpu(&self) -> Tensor<CpuBackend, 2> {
            let cx = 0.5 + 0.25 * (self.time * 0.7).cos();
            let cy = 0.5 + 0.25 * (self.time * 0.9).sin();
            let k = 20.0; let omega = 6.0; let damping = 1.5; let amp = 1.0;
            let dx = &self.coords_cpu_x - cx;
            let dy = &self.coords_cpu_y - cy;
            let r = (dx.clone() * dx + dy.clone() * dy).sqrt();
            let phase = r.clone() * k - self.time * omega;
            let env = (r * -damping).exp();
            phase.sin() * env * amp
        }

        // Stateful ripple: simple persistence filter driven by analytic seed
        fn ripple_stateful_gpu(&mut self) -> Tensor<GpuBackend, 2> {
            let drive = self.ripple_analytic_gpu();
            // u = 0.97*u + 0.03*drive (device-side ops)
            self.ripple_state_gpu = self.ripple_state_gpu.clone() * 0.97 + drive * 0.03;
            self.ripple_state_gpu.clone()
        }
        fn ripple_stateful_cpu(&mut self) -> Tensor<CpuBackend, 2> {
            let drive = self.ripple_analytic_cpu();
            self.ripple_state_cpu = self.ripple_state_cpu.clone() * 0.97 + drive * 0.03;
            self.ripple_state_cpu.clone()
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
                // Feedback source from previous frame
                let feedback = if i == 0 { self.grid_tensors_gpu[5].clone() } else { self.grid_tensors_gpu[i - 1].clone() };
                let t = match i {
                    // T1 = gain_ripple * ripple + gain_feedback * feedback
                    0 => {
                        let ripple = if self.ripple_mode == RippleMode::Analytic { self.ripple_analytic_gpu() } else { self.ripple_stateful_gpu() };
                        ripple * self.gain_ripple + feedback * self.gain_feedback
                    }
                    1 => feedback * 0.8 + 0.2,
                    2 => feedback.clamp_min(0.2).clamp_max(0.8),
                    3 => feedback * -1.0 + 1.0,
                    4 => feedback + Tensor::<GpuBackend, 2>::random([GRID_TENSOR_SIZE, GRID_TENSOR_SIZE], Distribution::Default, device) * 0.1,
                    5 => feedback * ((self.time as f32).sin() + 1.1),
                    _ => feedback,
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
                let feedback = if i == 0 { self.grid_tensors_cpu[5].clone() } else { self.grid_tensors_cpu[i - 1].clone() };
                let t = match i {
                    0 => {
                        let ripple = if self.ripple_mode == RippleMode::Analytic { self.ripple_analytic_cpu() } else { self.ripple_stateful_cpu() };
                        ripple * self.gain_ripple + feedback * self.gain_feedback
                    }
                    1 => feedback * 0.8 + 0.2,
                    2 => feedback.clamp_min(0.2).clamp_max(0.8),
                    3 => feedback * -1.0 + 1.0,
                    4 => feedback + Tensor::<CpuBackend, 2>::random([GRID_TENSOR_SIZE, GRID_TENSOR_SIZE], Distribution::Default, device) * 0.1,
                    5 => feedback * ((self.time as f32).sin() + 1.1),
                    _ => feedback,
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
            // Ripple mode toggle
            if self.window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
                self.ripple_mode = if self.ripple_mode == RippleMode::Analytic { RippleMode::Stateful } else { RippleMode::Analytic };
                println!("Ripple Mode: {:?}", self.ripple_mode);
            }
            // Gains
            if self.window.is_key_down(Key::Z) { self.gain_ripple = (self.gain_ripple - 0.01).max(0.0); }
            if self.window.is_key_down(Key::X) { self.gain_ripple = (self.gain_ripple + 0.01).min(2.0); }
            if self.window.is_key_down(Key::C) { self.gain_feedback = (self.gain_feedback - 0.01).max(0.0); }
            if self.window.is_key_down(Key::V) { self.gain_feedback = (self.gain_feedback + 0.01).min(2.0); }
            // Controls similar to previous impl
            if self.window.is_key_down(Key::Up) { self.animation_speed = (self.animation_speed * 1.05).min(5.0); }
            if self.window.is_key_down(Key::Down) { self.animation_speed = (self.animation_speed * 0.95).max(0.1); }
            if self.window.is_key_down(Key::Left) { self.intensity_multiplier = (self.intensity_multiplier * 0.98).max(0.1); }
            if self.window.is_key_down(Key::Right) { self.intensity_multiplier = (self.intensity_multiplier * 1.02).min(3.0); }
            if self.window.is_key_down(Key::PageUp) { self.wave_complexity = (self.wave_complexity * 1.02).min(3.0); }
            if self.window.is_key_down(Key::PageDown) { self.wave_complexity = (self.wave_complexity * 0.98).max(0.1); }
            if self.window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) { self.print_status(); self.print_help(); }
            if self.window.is_key_pressed(Key::H, minifb::KeyRepeat::No) { self.print_help(); }
        }

        fn print_status(&self) {
            println!("\n🌀 6-Tensor Pipeline Status 🌀");
            println!("Backend: {}", if self.use_gpu { "GPU (WGPU)" } else { "CPU (NdArray)" });
            println!("Ripple Mode: {:?}", self.ripple_mode);
            println!("Gains: ripple={:.2}, feedback={:.2}", self.gain_ripple, self.gain_feedback);
            println!("Animation Speed: {:.2}x", self.animation_speed);
            println!("Intensity: {:.2}", self.intensity_multiplier);
            println!("Wave Complexity: {:.2}", self.wave_complexity);
        }

        fn print_help(&self) {
            println!("\nKeys:");
            println!("  Esc        : Exit");
            println!("  G          : Toggle backend (CPU/NdArray ↔ GPU/WGPU)");
            println!("  R          : Toggle ripple mode (Analytic ↔ Stateful)");
            println!("  Z / X      : Decrease / Increase ripple gain");
            println!("  C / V      : Decrease / Increase feedback gain");
            println!("  ↑ / ↓      : Animation speed ±");
            println!("  ← / →      : Intensity ±");
            println!("  PgUp/PgDn  : Wave complexity ±");
            println!("  Space / H  : Print status/help");
            println!("Display: 3x2 panes (T1..T6 left→right, top→bottom); T6 feeds back into T1.");
        }

        pub fn run_loop(&mut self) {
            println!("🔥🌀 Starting Burn 6-Tensor Pipeline Visualizer 🌀🔥");
            println!("Ready. Press G to switch backend, Esc to exit.");
            self.print_status();
            self.print_help();
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
