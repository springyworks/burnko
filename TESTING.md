# Test Organization in Burn Workspace

This document explains how tests are organized in the Burn workspace and where to place different types of tests.

## Test Directory Structure

```
burnko/
├── tests/                          # Workspace-level integration tests
│   ├── cross_backend_comparison.rs # CPU vs GPU performance comparisons
│   └── scan_verification.rs        # Multi-backend verification tests
├── workspace-tests/                # Development and experimental tests
│   ├── debug_axis.rs               # Axis debugging utilities
│   ├── fft_test_simple.rs          # Simple FFT tests
│   ├── heavy_lifting_test.rs       # Performance benchmarks
│   └── ...                        # Other development tests
├── crates/
│   ├── burn-ndarray/
│   │   └── tests/                  # NdArray-specific tests
│   │       ├── scan_verification.rs
│   │       └── fft_test.rs
│   ├── burn-wgpu/
│   │   └── tests/                  # WGPU-specific tests
│   │       ├── gpu_burn_monitor.rs
│   │       └── gpu_2d_performance.rs
│   └── burn-cubecl/
│       └── tests/                  # CubeCL kernel tests
└── examples/                       # Complete examples and demos
    ├── realtime-tensor-viz/        # Interactive visualization
    └── debug_cumsum_precision/     # Backend comparison examples
```

## Where to Place Tests

### 1. Workspace Root `/tests/` Directory
✅ **Use for**: Cross-crate integration tests
- Tests that compare multiple backends (NdArray vs WGPU)
- End-to-end functionality tests
- Performance comparisons across backends
- Integration tests that span multiple crates

**Example**: `cross_backend_comparison.rs` - compares CPU vs GPU performance

### 2. Crate-Specific `/crates/*/tests/` Directories  
✅ **Use for**: Backend-specific tests
- Tests that focus on a single backend implementation
- Kernel-level tests for GPU operations
- Backend-specific performance optimizations
- Unit tests for backend features

**Examples**: 
- `burn-ndarray/tests/` - CPU implementation tests
- `burn-wgpu/tests/` - GPU implementation tests
- `burn-cubecl/tests/` - Compute shader tests

### 3. Workspace `/workspace-tests/` Directory
✅ **Use for**: Development and experimental tests
- Temporary test files during development
- Debugging utilities and helpers
- Experimental performance tests
- Quick prototypes and verification scripts

### 4. `/examples/` Directory
✅ **Use for**: Complete applications and demos
- Interactive examples showing functionality
- Documentation examples
- Performance demonstrations
- User-facing sample applications

## Test Naming Conventions

### Integration Tests (workspace `/tests/`)
```rust
// tests/cross_backend_comparison.rs
fn test_cpu_vs_gpu_scan_performance()
fn test_cpu_vs_gpu_fft_performance()
```

### Backend-Specific Tests  
```rust
// crates/burn-wgpu/tests/gpu_performance.rs
fn test_gpu_scan_2d_performance()
fn test_gpu_memory_utilization()

// crates/burn-ndarray/tests/cpu_optimization.rs  
fn test_parallel_scan_performance()
fn test_large_tensor_handling()
```

### Development Tests
```rust
// workspace-tests/debug_*.rs
fn debug_axis_iteration()
fn verify_tensor_layout()
```

## Running Tests

```bash
# Run all workspace tests
cargo test

# Run workspace integration tests only
cargo test --test cross_backend_comparison

# Run specific crate tests
cargo test -p burn-wgpu
cargo test -p burn-ndarray

# Run specific test file
cargo test --test gpu_2d_performance

# Run with features
cargo test --features std

# Run ignored tests (like performance benchmarks)
cargo test -- --include-ignored
```

## Best Practices

1. **Cross-Backend Tests**: Put in workspace `/tests/` when comparing backends
2. **Performance Tests**: Use `#[ignore]` for expensive tests, enable with `--include-ignored`
3. **GPU Tests**: Always check for GPU availability before running
4. **Documentation**: Include performance expectations and test purposes in comments
5. **Organization**: Group related tests in the same file, use descriptive names

This organization ensures tests are easy to find, run, and maintain while providing clear separation between different types of testing needs.
