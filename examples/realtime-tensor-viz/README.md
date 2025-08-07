# Real-time Tensor Visualization

This example provides dynamic visualization of 2D tensors and their FFT/scan operations in real-time, supporting both NdArray (CPU) and WGPU (GPU) backends.

## Features

🎨 **Live Visualization**
- Real-time 2D tensor heatmaps
- Dynamic FFT magnitude visualization  
- Scan operation results display
- Animated test patterns

⚡ **Performance Comparison**
- CPU vs GPU backend switching
- Real-time performance metrics
- Throughput measurements (Melems/sec)
- Timing comparisons

🎮 **Interactive Controls**
- SPACE: Toggle between CPU and GPU backends
- 1-4: Switch between scan operations (cumsum, cumprod, cummax, cummin)
- Q: Quit application

## Usage

```bash
cd examples/realtime-tensor-viz
cargo run
```

## Display Layout

The window shows three panels:
1. **Original Tensor** - The input 2D tensor with animated patterns
2. **Scan Result** - Result of the selected scan operation  
3. **FFT Magnitude** - Magnitude spectrum of the FFT

## Performance Metrics

The bottom of the window displays:
- Current backend (CPU/GPU)
- Selected operation type
- CPU performance (time and throughput)
- GPU performance (time and throughput)

## Dependencies

- `burn` - Main tensor framework
- `burn-ndarray` - CPU backend
- `burn-wgpu` - GPU backend  
- `minifb` - Window management and display
- `rayon` - Parallel processing

## Technical Details

- Tensor size: 256×256 elements
- Update rate: ~20 FPS
- Color mapping: Blue (low) → Cyan → Green → Yellow → Red (high)
- Real-time pattern animation for visual feedback

This example demonstrates the power and flexibility of the Burn tensor framework for high-performance computing with beautiful real-time visualization!
