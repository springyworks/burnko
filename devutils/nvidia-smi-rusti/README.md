# nvidia-smi-rusti

A Rust utility for monitoring and debugging Burn (Rust) processes on NVIDIA GPUs using `nvidia-smi`.

## Features
- List all processes using the GPU, with PID, process name, memory usage, and GPU UUID
- Filter by process name (e.g. `burn`, `rust`, or your binary name)
- Filter by PID
- Show all GPU processes (ignore filters)

## Usage

```sh
cargo run --release -- [OPTIONS]
```

### Options
- `-n`, `--name <NAME>`: Filter by process name (regex, default: "")
- `-p`, `--pid <PID>`: Filter by PID
- `--all`: Show all GPU processes (ignore filters)

### Examples

Show all GPU processes:
```sh
cargo run --release -- --all
```

Show only Burn or Rust processes:
```sh
cargo run --release -- -n burn
```

Show only a specific PID:
```sh
cargo run --release -- -p 12345
```

## Integration
You can call this utility from your Burn project, or use it as a standalone tool for debugging and monitoring GPU usage.
