# Cross-Backend Comparison Example

This example demonstrates how to:

1. **Use Burn's testgen system** - The proper way to write cross-backend tests in Burn
2. **Compare performance** between NdArray (CPU) and WGPU (GPU) backends  
3. **Verify correctness** across different backends
4. **Benchmark scan and FFT operations** with real performance metrics

## Burn's Test Management System

Burn uses a sophisticated **burn-tensor-testgen** system that automatically generates tests across all backends. Instead of manually writing cross-backend tests, you should use:

```rust
#[burn_tensor_testgen::testgen(operation_name)]
mod tests {
    // Your tests here - they automatically run on all backends
}
```

### Key Benefits:

- ✅ **Automatic cross-backend testing** - No manual backend switching needed
- ✅ **Consistent test coverage** - All backends get the same tests
- ✅ **Type safety** - Uses `TestTensor` that works with any backend
- ✅ **Integrated with CI** - Tests run automatically across all supported backends

## Running the Example

```bash
# Basic comparison
cargo run --bin main

# With specific features
cargo run --bin main --features "wgpu"

# Run tests
cargo test
```

## Output Example

```
🔥 Burn Cross-Backend Testing Framework
=====================================

=== Cross-Backend Scan Operation Benchmarks ===

🔄 Testing NdArray backend:
  ✅ Cumulative sum: 0.25ms avg

🚀 Testing WGPU backend:
  ✅ Cumulative sum: 0.05ms avg

📊 Performance Comparison:
  WGPU is 5.00x faster than NdArray

=== Cross-Backend FFT Operation Benchmarks ===

🔄 Testing NdArray backend:
  ✅ 2D FFT: 2.50ms avg

🚀 Testing WGPU backend:
  ✅ 2D FFT: 0.51ms avg
  WGPU throughput: 12.34 GB/s

📊 Performance Comparison:
  WGPU is 4.90x faster than NdArray

=== Cross-Backend Correctness Verification ===
🔄 NdArray cumsum computed (32 elements)
🚀 WGPU cumsum computed (32 elements)
  ✅ Perfect match! Max difference: 1.23e-07

🎯 Testing completed!
```

## Integration with Burn's testgen System

For proper cross-backend tests in Burn, add your tests to:
- `crates/burn-tensor/src/tests/ops/your_operation.rs`
- Use `#[burn_tensor_testgen::testgen(your_operation)]`
- Add `burn_tensor::testgen_your_operation!();` to `crates/burn-tensor/src/tests/mod.rs`

This ensures your tests run automatically across ALL Burn backends (NdArray, WGPU, Candle, LibTorch, etc.).
