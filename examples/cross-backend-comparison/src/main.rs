use burn::{
    backend::{ndarray::{NdArray, NdArrayDevice}, wgpu::{Wgpu, WgpuDevice}},
    tensor::Tensor,
};

// Type aliases for clarity
type CpuBackend = NdArray<f32>;
type GpuBackend = Wgpu<f32>;

struct BenchCfg {
    size: usize,
    iterations: usize,
}

impl Default for BenchCfg {
    fn default() -> Self { Self { size: 1_048_576, iterations: 50 } }
}

fn benchmark_scan_cumsum() {
    println!("=== Cross-Backend Cumsum Benchmarks ===");
    let cfg = BenchCfg::default();

    let data: Vec<f32> = (0..cfg.size).map(|i| (i as f32).sin() + 1.0).collect();

    // CPU (NdArray)
    let cpu_dev = NdArrayDevice::default();
    let cpu_tensor: Tensor<CpuBackend, 1> = Tensor::from_floats(data.as_slice(), &cpu_dev);
    let cpu_start = std::time::Instant::now();
    for _ in 0..cfg.iterations { let _ = cpu_tensor.clone().cumsum(0); }
    let cpu_dur = cpu_start.elapsed();

    // GPU (WGPU)
    let gpu_dev = WgpuDevice::default();
    let gpu_tensor: Tensor<GpuBackend, 1> = Tensor::from_floats(data.as_slice(), &gpu_dev);
    let gpu_start = std::time::Instant::now();
    for _ in 0..cfg.iterations { let _ = gpu_tensor.clone().cumsum(0); }
    let gpu_dur = gpu_start.elapsed();

    let cpu_avg_ms = cpu_dur.as_secs_f64() * 1000.0 / cfg.iterations as f64;
    let gpu_avg_ms = gpu_dur.as_secs_f64() * 1000.0 / cfg.iterations as f64;

    println!("CPU (NdArray) avg: {:.3} ms", cpu_avg_ms);
    println!("GPU (WGPU)   avg: {:.3} ms", gpu_avg_ms);

    if gpu_avg_ms > 0.0 {
        let speedup = cpu_avg_ms / gpu_avg_ms;
        println!("Speedup GPU/CPU: {:.2}x", speedup);
    }
}

fn verify_cumsum_correctness() {
    println!("\n=== Correctness: cumsum across backends (tolerant) ===");
    let size = 2048;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.25 + 0.1).collect();

    let cpu_dev = NdArrayDevice::default();
    let gpu_dev = WgpuDevice::default();

    let r_cpu = Tensor::<CpuBackend, 1>::from_floats(data.as_slice(), &cpu_dev)
        .cumsum(0)
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let r_gpu = Tensor::<GpuBackend, 1>::from_floats(data.as_slice(), &gpu_dev)
        .cumsum(0)
        .to_data()
        .to_vec::<f32>()
        .unwrap();

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut mismatches = 0usize;
    for (i, (a, b)) in r_cpu.iter().zip(r_gpu.iter()).enumerate() {
        let diff = (a - b).abs();
        max_abs = max_abs.max(diff);
        let scale = a.abs().max(b.abs()).max(1.0);
        let rel = diff / scale;
        max_rel = max_rel.max(rel);
        if rel > 2e-5 { // small tolerance for float associativity
            if mismatches < 10 {
                println!("Mismatch at {}: CPU={:.9}, GPU={:.9}, rel={:.3e}", i, a, b, rel);
            }
            mismatches += 1;
        }
    }

    println!("Max abs diff: {:.3e}", max_abs);
    println!("Max rel diff: {:.3e}", max_rel);
    assert_eq!(mismatches, 0, "Cumsum mismatch above tolerance");
}

fn main() {
    println!("🔥 Burn Cross-Backend: Scan/Cumsum");
    benchmark_scan_cumsum();
    verify_cumsum_correctness();
    println!("\n🎯 Done");
}
