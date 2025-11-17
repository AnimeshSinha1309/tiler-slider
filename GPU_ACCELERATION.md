# GPU Acceleration Guide

The tiler-slider neural network models now support GPU acceleration for dramatically faster training.

## Supported Accelerators

The system automatically detects and uses the best available accelerator in this order:

1. **NVIDIA CUDA** (Linux/Windows with NVIDIA GPU)
2. **Apple Metal (MPS)** (macOS with Apple Silicon)
3. **CPU** (Fallback)

## Automatic Device Detection

By default, the training script automatically detects and uses the best available device:

```bash
# Auto-detect (recommended)
uv run python -m explainrl.models.train --num-puzzles 20 --multi-color
```

## Manual Device Selection

You can override the automatic detection:

```bash
# Force CPU
uv run python -m explainrl.models.train --no-gpu --num-puzzles 20

# Use CUDA explicitly
uv run python -m explainrl.models.train --device cuda --num-puzzles 20

# Use Metal/MPS explicitly (macOS)
uv run python -m explainrl.models.train --device mps --num-puzzles 20

# Use CPU explicitly
uv run python -m explainrl.models.train --device cpu --num-puzzles 20
```

## Device Information

Check what devices are available on your system:

```python
from explainrl.models import print_device_info

print_device_info()
```

Output example (Linux with NVIDIA GPU):
```
============================================================
Device Information
============================================================
Platform: Linux
Processor: x86_64
CPU Threads: 16

CUDA (NVIDIA GPU): Available ✓
  - NVIDIA GeForce RTX 3090
    Compute Capability: (8, 6)
    Total Memory: 24.0 GB

MPS (Apple Silicon GPU): Not available
============================================================
```

Output example (macOS with Apple Silicon):
```
============================================================
Device Information
============================================================
Platform: Darwin
Processor: arm
CPU Threads: 8

CUDA (NVIDIA GPU): Not available

MPS (Apple Silicon GPU): Available ✓
============================================================
```

## Performance Comparison

Typical speedups compared to CPU training:

| Device | Speedup | Example Timing (100 forward passes) |
|--------|---------|--------------------------------------|
| CPU (x86) | 1x | ~10ms per pass (~93 passes/sec) |
| Apple M1/M2 (MPS) | 10-20x | ~0.5-1ms per pass (~1000-2000 passes/sec) |
| NVIDIA RTX 3090 (CUDA) | 50-100x | ~0.1-0.2ms per pass (~5000-10000 passes/sec) |

**Note**: Actual speedup depends on:
- Model size (hidden_dim, num_recursive_steps)
- Batch size
- Board size
- GPU memory bandwidth
- CPU vs GPU memory transfer overhead

## Using GPU in Code

```python
from explainrl.models import TilerSliderNet, PPOTrainer, get_device
from explainrl.environment import TilerSliderEnv

# Auto-detect device
device = get_device()
print(f"Using device: {device}")

# Create model (will be moved to device automatically)
model = TilerSliderNet(hidden_dim=256, num_recursive_steps=3)

# Create trainer (auto-detects device)
trainer = PPOTrainer(model, device=None)  # None = auto-detect

# Or explicitly specify device
trainer = PPOTrainer(model, device='cuda')  # Use CUDA
trainer = PPOTrainer(model, device='mps')   # Use Metal/MPS
trainer = PPOTrainer(model, device='cpu')   # Use CPU
```

## Memory Considerations

### GPU Memory Requirements

Approximate GPU memory needed for different configurations:

| Hidden Dim | Recursive Steps | Batch Size | GPU Memory |
|------------|----------------|------------|------------|
| 128 | 2 | 32 | ~0.5 GB |
| 256 | 3 | 64 | ~1.5 GB |
| 512 | 5 | 128 | ~4 GB |
| 1024 | 7 | 256 | ~12 GB |

### Out of Memory Errors

If you get CUDA OOM (Out Of Memory) errors:

1. **Reduce batch size**:
   ```bash
   --batch-size 32  # Instead of 64
   ```

2. **Reduce hidden dimension**:
   ```bash
   --hidden-dim 128  # Instead of 256
   ```

3. **Reduce recursive steps**:
   ```bash
   --num-recursive-steps 2  # Instead of 3
   ```

4. **Reduce trajectories per environment**:
   ```bash
   --trajectories-per-env 5  # Instead of 10
   ```

5. **Use gradient accumulation** (advanced):
   Collect smaller batches and accumulate gradients

### Apple Silicon (M1/M2/M3) Notes

- MPS is well-optimized for transformer-style models (like our TRM)
- Unified memory architecture means less overhead
- Memory is shared with system, so close other apps for best performance
- Some PyTorch operations may fall back to CPU (automatically)

### CUDA Notes

- Use `nvidia-smi` to monitor GPU utilization and memory
- Enable mixed precision training for faster training (future feature)
- Consider using multiple GPUs with DataParallel (future feature)

## Benchmarking

Run the included benchmark script to test performance on your hardware:

```python
#!/usr/bin/env python3
"""Benchmark GPU acceleration."""
import time
import torch
from explainrl.models import TilerSliderNet, get_device, print_device_info
from explainrl.environment import TilerSliderEnv

print_device_info()

device = get_device()
model = TilerSliderNet(hidden_dim=256, num_recursive_steps=3)
model = model.to(device)

env = TilerSliderEnv(size=4, initial_locations=[(0,0)], target_locations=[(3,3)])
env.reset()

# Warm-up
for _ in range(10):
    model.forward(env.state)

# Benchmark
num_runs = 1000
start = time.time()
for _ in range(num_runs):
    value, policy, _ = model.forward(env.state)
    if device == 'cuda':
        torch.cuda.synchronize()

elapsed = time.time() - start
print(f"\n{num_runs} forward passes in {elapsed:.2f}s")
print(f"Throughput: {num_runs/elapsed:.0f} passes/sec")
```

## Troubleshooting

### CUDA Not Detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"
```

If CUDA is not available:
- Install NVIDIA drivers
- Install CUDA toolkit
- Reinstall PyTorch with CUDA support:
  ```bash
  uv pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```

### MPS Not Working (macOS)

```bash
# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"
```

If MPS is not available:
- Update to macOS 12.3 or later
- Update to PyTorch 1.12 or later
- Use Apple Silicon Mac (M1/M2/M3)

### Slower Than Expected

Common causes:
1. **Small batch sizes**: GPU has overhead, needs larger batches to be efficient
2. **CPU-GPU transfer**: Minimize data movement between CPU and GPU
3. **Synchronization**: Avoid unnecessary `torch.cuda.synchronize()` calls
4. **Background processes**: Close other GPU-using applications

## Best Practices

### For Training Speed

1. **Use the largest batch size that fits in memory**
   - Start with 64, increase until OOM
   - Use `--batch-size 128` or `--batch-size 256`

2. **Collect more trajectories per iteration**
   - More parallelism with GPU
   - Use `--trajectories-per-env 20` instead of 10

3. **Use larger models**
   - GPU shines with bigger models
   - Try `--hidden-dim 512` or `--hidden-dim 1024`

4. **Enable mixed precision** (future)
   - 2x faster training
   - Half the memory usage

### For Memory Efficiency

1. **Use smaller models for prototyping**
   - `--hidden-dim 64` for quick tests
   - `--num-recursive-steps 1` for debugging

2. **Clear cache between runs**
   ```python
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
   ```

3. **Monitor memory usage**
   ```python
   print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
   ```

## Examples

### Quick Test (CPU)
```bash
uv run python -m explainrl.models.train \
    --no-gpu \
    --num-puzzles 5 \
    --num-iterations 10 \
    --hidden-dim 64
```

### Fast Training (GPU)
```bash
uv run python -m explainrl.models.train \
    --num-puzzles 20 \
    --multi-color \
    --num-iterations 1000 \
    --hidden-dim 512 \
    --num-recursive-steps 5 \
    --batch-size 128 \
    --trajectories-per-env 20
```

### Large-Scale Training (Multi-GPU, future)
```bash
uv run python -m explainrl.models.train \
    --num-puzzles 50 \
    --multi-color \
    --num-iterations 10000 \
    --hidden-dim 1024 \
    --num-recursive-steps 7 \
    --batch-size 256 \
    --trajectories-per-env 50 \
    --device cuda
```

## Future Enhancements

Planned optimizations:
- [ ] Mixed precision training (AMP)
- [ ] Multi-GPU training (DataParallel/DistributedDataParallel)
- [ ] Gradient accumulation for larger effective batch sizes
- [ ] Model parallelism for very large models
- [ ] Optimized kernels for specific operations
- [ ] Batch inference for trajectory collection
