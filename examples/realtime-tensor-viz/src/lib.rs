//! Realtime Tensor Visualizer with a 6-tensor closed feedback grid.

mod viz {
    use burn::{
        tensor::Tensor,
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
    const GRID_TENSOR_SIZE: usize = 128; // 128x128 tensors for grid view
    const CELL_WIDTH: usize = (WINDOW_WIDTH - (GRID_COLS + 1) * CELL_MARGIN) / GRID_COLS;
    const CELL_HEIGHT: usize = (WINDOW_HEIGHT - (GRID_ROWS + 1) * CELL_MARGIN) / GRID_ROWS;

    type CpuBackend = NdArray<f32>;
    type GpuBackend = Wgpu<f32>;
    type CpuDevice = NdArrayDevice;
    type GpuDevice = WgpuDevice;

    #[derive(Clone, Copy, Debug)]
    enum VisualizationMode { PsychedelicWaves, CosmicSpiral, TensorStorm, PlasmaField, QuantumRipples, HypnoticMandalas }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum RippleMode { Analytic, Stateful }

    #[derive(Clone)]
    struct PerformanceStats { times: Vec<Duration> }
    impl PerformanceStats { fn new() -> Self { Self { times: Vec::new() } } }

    pub struct Visualizer {
        window: Option<Window>,
        buffer: Vec<u32>,
        cpu_device: CpuDevice,
        gpu_device: GpuDevice,
        time: f32,
        pub animation_speed: f32,
        visualization_mode: VisualizationMode,
        pub use_gpu: bool,
        color_phase: f32,
        pub intensity_multiplier: f32,
        pub wave_complexity: f32,
        performance_stats: Arc<Mutex<PerformanceStats>>,
        grid_tensors_gpu: Vec<Tensor<GpuBackend, 2>>,
        grid_tensors_cpu: Vec<Tensor<CpuBackend, 2>>,
        ripple_mode: RippleMode,
        pub gain_ripple: f32,
        pub gain_feedback: f32,
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
            let zero_gpu = Tensor::<GpuBackend, 2>::zeros([GRID_TENSOR_SIZE, GRID_TENSOR_SIZE], &gpu_device);
            let zero_cpu = Tensor::<CpuBackend, 2>::zeros([GRID_TENSOR_SIZE, GRID_TENSOR_SIZE], &cpu_device);

            Self {
                window: Some(window),
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

        // New: headless constructor (no minifb window)
        pub fn new_headless() -> Self {
            let buffer = vec![0; WINDOW_WIDTH * WINDOW_HEIGHT];
            let cpu_device = CpuDevice::default();
            let gpu_device = GpuDevice::default();
            use burn::tensor::Distribution;
            let mut grid_tensors_gpu = Vec::with_capacity(6);
            let t0_gpu: Tensor<GpuBackend, 2> = Tensor::random([GRID_TENSOR_SIZE, GRID_TENSOR_SIZE], Distribution::Default, &gpu_device);
            for _ in 0..6 { grid_tensors_gpu.push(t0_gpu.clone()); }
            let mut grid_tensors_cpu = Vec::with_capacity(6);
            let t0_cpu: Tensor<CpuBackend, 2> = Tensor::random([GRID_TENSOR_SIZE, GRID_TENSOR_SIZE], Distribution::Default, &cpu_device);
            for _ in 0..6 { grid_tensors_cpu.push(t0_cpu.clone()); }
            let (coords_gpu_x, coords_gpu_y) = Self::build_coords_gpu(&gpu_device);
            let (coords_cpu_x, coords_cpu_y) = Self::build_coords_cpu(&cpu_device);
            let zero_gpu = Tensor::<GpuBackend, 2>::zeros([GRID_TENSOR_SIZE, GRID_TENSOR_SIZE], &gpu_device);
            let zero_cpu = Tensor::<CpuBackend, 2>::zeros([GRID_TENSOR_SIZE, GRID_TENSOR_SIZE], &cpu_device);
            Self {
                window: None,
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

        // Public: one simulation step without drawing to minifb
        pub fn step_frame(&mut self) {
            self.update_animation_state();
            if self.use_gpu { self.update_grid_pipeline_gpu(); } else { self.update_grid_pipeline_cpu(); }
        }

        // Public: get grid size
        pub fn grid_tensor_size(&self) -> usize { GRID_TENSOR_SIZE }

        // Public: export 6 RGBA images (w,h,bytes)
        pub fn export_grid_rgba(&self) -> Vec<(usize, usize, Vec<u8>)> {
            let tensors: Vec<Vec<f32>> = if self.use_gpu {
                self.grid_tensors_gpu.iter().map(|t| t.to_data().as_slice().unwrap().to_vec()).collect()
            } else {
                self.grid_tensors_cpu.iter().map(|t| t.to_data().as_slice().unwrap().to_vec()).collect()
            };
            let mut out = Vec::with_capacity(6);
            for data in tensors {
                let mut bytes = Vec::with_capacity(GRID_TENSOR_SIZE * GRID_TENSOR_SIZE * 4);
                for &v in &data { let [r,g,b,a] = self.f32_to_rgba(v); bytes.extend_from_slice(&[r,g,b,a]); }
                out.push((GRID_TENSOR_SIZE, GRID_TENSOR_SIZE, bytes));
            }
            out
        }

        fn build_coords_gpu(device: &GpuDevice) -> (Tensor<GpuBackend, 2>, Tensor<GpuBackend, 2>) {
            let w = GRID_TENSOR_SIZE; let h = GRID_TENSOR_SIZE;
            let mut xs = Vec::with_capacity(w * h);
            let mut ys = Vec::with_capacity(w * h);
            for j in 0..h { for i in 0..w { xs.push(i as f32 / (w - 1) as f32); ys.push(j as f32 / (h - 1) as f32); } }
            let x = Tensor::<GpuBackend, 1>::from_floats(xs.as_slice(), device).reshape([h, w]);
            let y = Tensor::<GpuBackend, 1>::from_floats(ys.as_slice(), device).reshape([h, w]);
            (x, y)
        }
        fn build_coords_cpu(device: &CpuDevice) -> (Tensor<CpuBackend, 2>, Tensor<CpuBackend, 2>) {
            let w = GRID_TENSOR_SIZE; let h = GRID_TENSOR_SIZE;
            let mut xs = Vec::with_capacity(w * h);
            let mut ys = Vec::with_capacity(w * h);
            for j in 0..h { for i in 0..w { xs.push(i as f32 / (w - 1) as f32); ys.push(j as f32 / (h - 1) as f32); } }
            let x = Tensor::<CpuBackend, 1>::from_floats(xs.as_slice(), device).reshape([h, w]);
            let y = Tensor::<CpuBackend, 1>::from_floats(ys.as_slice(), device).reshape([h, w]);
            (x, y)
        }

        // Color mapping helpers
        fn f32_to_rgba(&self, value: f32) -> [u8;4] {
            let normalized = (value.tanh() + 1.0) * 0.5; // [0,1]
            let phase_r = self.color_phase;
            let phase_g = self.color_phase + 2.0943; // 120°
            let phase_b = self.color_phase + 4.1888; // 240°
            let r = ((normalized * 6.28 + phase_r).sin() * 0.5 + 0.5) * 255.0;
            let g = ((normalized * 6.28 + phase_g).sin() * 0.5 + 0.5) * 255.0;
            let b = ((normalized * 6.28 + phase_b).sin() * 0.5 + 0.5) * 255.0;
            let intensity = self.intensity_multiplier * (1.0 + (self.time * 4.0).sin() * 0.2);
            let r = (r * intensity).clamp(0.0, 255.0) as u8;
            let g = (g * intensity).clamp(0.0, 255.0) as u8;
            let b = (b * intensity).clamp(0.0, 255.0) as u8;
            [r,g,b,255]
        }
        fn tensor_to_color(&self, value: f32) -> u32 {
            let [r,g,b,a] = self.f32_to_rgba(value);
            ((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
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

        // Minifb rendering path
        fn render_six_tensor_grid(&mut self) {
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
            if let Some(window) = &mut self.window {
                window.update_with_buffer(&self.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
                    .expect("Failed to update window");
            }
        }

        fn ripple_analytic_gpu(&self) -> Tensor<GpuBackend, 2> {
            let cx = 0.5 + 0.25 * (self.time * 0.7).cos();
            let cy = 0.5 + 0.25 * (self.time * 0.9).sin();
            let k = 20.0; let omega = 6.0; let damping = 1.5; let amp = 1.0;
            let dx = self.coords_gpu_x.clone().sub_scalar(cx);
            let dy = self.coords_gpu_y.clone().sub_scalar(cy);
            let r = (dx.clone() * dx + dy.clone() * dy).sqrt();
            let phase = r.clone().mul_scalar(k).sub_scalar(self.time * omega);
            let env = r.mul_scalar(-damping).exp();
            phase.sin() * env * amp
        }
        fn ripple_analytic_cpu(&self) -> Tensor<CpuBackend, 2> {
            let cx = 0.5 + 0.25 * (self.time * 0.7).cos();
            let cy = 0.5 + 0.25 * (self.time * 0.9).sin();
            let k = 20.0; let omega = 6.0; let damping = 1.5; let amp = 1.0;
            let dx = self.coords_cpu_x.clone().sub_scalar(cx);
            let dy = self.coords_cpu_y.clone().sub_scalar(cy);
            let r = (dx.clone() * dx + dy.clone() * dy).sqrt();
            let phase = r.clone().mul_scalar(k).sub_scalar(self.time * omega);
            let env = r.mul_scalar(-damping).exp();
            phase.sin() * env * amp
        }
        fn ripple_stateful_gpu(&mut self) -> Tensor<GpuBackend, 2> {
            let drive = self.ripple_analytic_gpu();
            self.ripple_state_gpu = self.ripple_state_gpu.clone().mul_scalar(0.97) + drive.mul_scalar(0.03);
            self.ripple_state_gpu.clone()
        }
        fn ripple_stateful_cpu(&mut self) -> Tensor<CpuBackend, 2> {
            let drive = self.ripple_analytic_cpu();
            self.ripple_state_cpu = self.ripple_state_cpu.clone().mul_scalar(0.97) + drive.mul_scalar(0.03);
            self.ripple_state_cpu.clone()
        }

        fn update_animation_state(&mut self) { self.time += 0.016 * self.animation_speed; self.color_phase += 0.02; }

        fn update_grid_pipeline_gpu(&mut self) {
            use burn::tensor::Distribution;
            let mut new_tensors = Vec::with_capacity(6);
            for i in 0..6 {
                let feedback = if i == 0 { self.grid_tensors_gpu[5].clone() } else { self.grid_tensors_gpu[i - 1].clone() };
                let t = match i {
                    0 => {
                        let ripple = if self.ripple_mode == RippleMode::Analytic { self.ripple_analytic_gpu() } else { self.ripple_stateful_gpu() };
                        ripple.mul_scalar(self.gain_ripple) + feedback.mul_scalar(self.gain_feedback)
                    }
                    1 => feedback * 0.8 + 0.2,
                    2 => feedback.clamp_min(0.2).clamp_max(0.8),
                    3 => feedback * -1.0 + 1.0,
                    4 => {
                        let noise = Tensor::<GpuBackend, 2>::random([GRID_TENSOR_SIZE, GRID_TENSOR_SIZE], Distribution::Default, &self.gpu_device);
                        feedback + noise.mul_scalar(0.1)
                    }
                    5 => feedback * ((self.time as f32).sin() + 1.1),
                    _ => feedback,
                };
                new_tensors.push(t);
            }
            self.grid_tensors_gpu = new_tensors;
        }
        fn update_grid_pipeline_cpu(&mut self) {
            use burn::tensor::Distribution;
            let mut new_tensors = Vec::with_capacity(6);
            for i in 0..6 {
                let feedback = if i == 0 { self.grid_tensors_cpu[5].clone() } else { self.grid_tensors_cpu[i - 1].clone() };
                let t = match i {
                    0 => {
                        let ripple = if self.ripple_mode == RippleMode::Analytic { self.ripple_analytic_cpu() } else { self.ripple_stateful_cpu() };
                        ripple.mul_scalar(self.gain_ripple) + feedback.mul_scalar(self.gain_feedback)
                    }
                    1 => feedback * 0.8 + 0.2,
                    2 => feedback.clamp_min(0.2).clamp_max(0.8),
                    3 => feedback * -1.0 + 1.0,
                    4 => {
                        let noise = Tensor::<CpuBackend, 2>::random([GRID_TENSOR_SIZE, GRID_TENSOR_SIZE], Distribution::Default, &self.cpu_device);
                        feedback + noise.mul_scalar(0.1)
                    }
                    5 => feedback * ((self.time as f32).sin() + 1.1),
                    _ => feedback,
                };
                new_tensors.push(t);
            }
            self.grid_tensors_cpu = new_tensors;
        }

        fn handle_input(&mut self) {
            let mut req_status = false;
            let mut req_help = false;
            if let Some(window) = &mut self.window {
                if window.is_key_down(Key::Escape) { std::process::exit(0); }
                if window.is_key_pressed(Key::G, minifb::KeyRepeat::No) {
                    self.use_gpu = !self.use_gpu;
                    println!("Backend: {}", if self.use_gpu { "GPU (WGPU)" } else { "CPU (NdArray)" });
                }
                if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
                    self.ripple_mode = if self.ripple_mode == RippleMode::Analytic { RippleMode::Stateful } else { RippleMode::Analytic };
                    println!("Ripple Mode: {:?}", self.ripple_mode);
                }
                if window.is_key_down(Key::Z) { self.gain_ripple = (self.gain_ripple - 0.01).max(0.0); }
                if window.is_key_down(Key::X) { self.gain_ripple = (self.gain_ripple + 0.01).min(2.0); }
                if window.is_key_down(Key::C) { self.gain_feedback = (self.gain_feedback - 0.01).max(0.0); }
                if window.is_key_down(Key::V) { self.gain_feedback = (self.gain_feedback + 0.01).min(2.0); }
                if window.is_key_down(Key::Up) { self.animation_speed = (self.animation_speed * 1.05).min(5.0); }
                if window.is_key_down(Key::Down) { self.animation_speed = (self.animation_speed * 0.95).max(0.1); }
                if window.is_key_down(Key::Left) { self.intensity_multiplier = (self.intensity_multiplier * 0.98).max(0.1); }
                if window.is_key_down(Key::Right) { self.intensity_multiplier = (self.intensity_multiplier * 1.02).min(3.0); }
                if window.is_key_down(Key::PageUp) { self.wave_complexity = (self.wave_complexity * 1.02).min(3.0); }
                if window.is_key_down(Key::PageDown) { self.wave_complexity = (self.wave_complexity * 0.98).max(0.1); }
                if window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) { req_status = true; req_help = true; }
                if window.is_key_pressed(Key::H, minifb::KeyRepeat::No) { req_help = true; }
            }
            if req_status { self.print_status(); }
            if req_help { self.print_help(); }
        }

        fn print_status(&self) {
            println!("\n🌀 6-Tensor Pipeline Status 🌀");
            println!("Backend: {}", if self.use_gpu { "GPU (WGPU)" } else { "CPU (NdArray)" });
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
            while self.window.as_ref().map(|w| w.is_open() && !w.is_key_down(Key::Escape)).unwrap_or(false) {
                self.handle_input();
                self.update_animation_state();
                self.render_six_tensor_grid();
            }
            println!("✨ Bye!");
        }
    }

    pub fn run() {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2).max(2);
        let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global();
        let mut v = Visualizer::new();
        v.run_loop();
    }
}

pub fn run() { viz::run() }

// Experimental egui frontend
pub mod experimental {
    use super::viz::{Visualizer, WINDOW_HEIGHT, WINDOW_WIDTH};
    use eframe::egui;

    pub fn run_egui() {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size(egui::vec2(WINDOW_WIDTH as f32, WINDOW_HEIGHT as f32))
                .with_min_inner_size(egui::vec2(900.0, 600.0)),
            ..Default::default()
        };
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2).max(2);
        let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global();
        eframe::run_native(
            "Burn Tensor Viz (experimental egui)",
            options,
            Box::new(|_cc| Box::new(App::new())),
        ).expect("failed to start egui app");
    }

    struct App {
        viz: Visualizer,
        textures: Option<Vec<egui::TextureHandle>>, // 6 textures
    }

    impl App { fn new() -> Self { Self { viz: Visualizer::new_headless(), textures: None } } }

    impl eframe::App for App {
        fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
            egui::SidePanel::left("controls").resizable(true).default_width(220.0).show(ctx, |ui| {
                ui.heading("Controls");
                ui.separator();
                ui.checkbox(&mut self.viz.use_gpu, "Use GPU (WGPU)");
                ui.add(egui::Slider::new(&mut self.viz.gain_ripple, 0.0..=2.0).text("Ripple gain"));
                ui.add(egui::Slider::new(&mut self.viz.gain_feedback, 0.0..=2.0).text("Feedback gain"));
                ui.add(egui::Slider::new(&mut self.viz.animation_speed, 0.1..=5.0).text("Speed"));
                ui.add(egui::Slider::new(&mut self.viz.intensity_multiplier, 0.1..=3.0).text("Intensity"));
                ui.add(egui::Slider::new(&mut self.viz.wave_complexity, 0.1..=3.0).text("Complexity"));
            });

            // Step simulation and get images
            self.viz.step_frame();
            let images = self.viz.export_grid_rgba();

            // Init textures if needed
            if self.textures.is_none() {
                let mut texs = Vec::with_capacity(6);
                for (i, (w,h,bytes)) in images.iter().enumerate() {
                    let img = egui::ColorImage::from_rgba_unmultiplied([*w as usize, *h as usize], &bytes);
                    let handle = ctx.load_texture(format!("pane_tex_{}", i), img, egui::TextureOptions::LINEAR);
                    texs.push(handle);
                }
                self.textures = Some(texs);
            } else {
                let texs = self.textures.as_mut().unwrap();
                for i in 0..6 {
                    let (w,h,bytes) = &images[i];
                    let img = egui::ColorImage::from_rgba_unmultiplied([*w as usize, *h as usize], &bytes);
                    texs[i].set(img, egui::TextureOptions::LINEAR);
                }
            }

            egui::CentralPanel::default().show(ctx, |ui| {
                let texs = self.textures.as_ref().unwrap();
                let spacing = 8.0;
                let available_w = ui.available_width();
                let cell_w = ((available_w - spacing * 4.0) / 3.0).max(64.0);
                let cell_h = cell_w; // square cells
                egui::Grid::new("tensor_grid").num_columns(3).spacing([spacing, spacing]).show(ui, |ui| {
                    for row in 0..2 {
                        for col in 0..3 {
                            let idx = row * 3 + col;
                            ui.image((texs[idx].id(), egui::vec2(cell_w, cell_h)));
                        }
                        ui.end_row();
                    }
                });
            });

            ctx.request_repaint();
        }
    }
}
