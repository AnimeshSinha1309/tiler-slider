# Training Guide for Tiler-Slider Neural Network

This guide shows how to train the TilerSliderNet using PPO with the fixed implementation.

## Quick Start

### 1. Run Tests

```bash
# Test the fixed state encoder
uv run pytest tests/test_models.py::TestStateEncoder -v

# Test the neural network
uv run pytest tests/test_models.py::TestTilerSliderNet -v

# Test PPO training
uv run pytest tests/test_models.py::TestPPOTrainer -v

# Run all model tests
uv run pytest tests/test_models.py -v
```

### 2. Train on Simple Puzzles

```bash
# Quick training on a few puzzles (for testing)
uv run python -m explainrl.models.train \
    --num-puzzles 5 \
    --multi-color \
    --num-iterations 100 \
    --hidden-dim 128 \
    --num-recursive-steps 2 \
    --trajectories-per-env 5 \
    --eval-interval 10
```

### 3. Full Training Run

```bash
# Full training on 20 puzzles
uv run python -m explainrl.models.train \
    --num-puzzles 20 \
    --multi-color \
    --num-iterations 1000 \
    --hidden-dim 256 \
    --num-recursive-steps 3 \
    --trajectories-per-env 10 \
    --save-dir checkpoints/
```

## What Was Fixed

The original code had a **channel mismatch error**:
```
RuntimeError: expected input[1, 7, 16, 16] to have 31 channels, but got 7 channels
```

### The Problem

The `StateEncoder` was creating variable-sized tensors:
- Single-color mode: 4 channels (blocked + tiles + targets + distances)
- Multi-color with 2 tiles: 7 channels (blocked + 2 tiles + 2 targets + 2 distances)
- But the Conv2d layer expected a fixed 31 channels (1 + 3×10 for max_tiles=10)

### The Solution

Modified `StateEncoder.encode()` to **always output max_channels** (1 + 3×max_tiles):
- Channel 0: Blocked cells
- Channels 1-10: Current tile positions (one per tile, up to max_tiles)
- Channels 11-20: Target positions (one per tile, up to max_tiles)
- Channels 21-30: Distance features (one per tile, up to max_tiles)
- Unused channels are filled with zeros

This ensures:
✅ Consistent input shape for Conv2d layers
✅ Support for both single-color and multi-color modes
✅ Support for varying numbers of tiles (1 to max_tiles)

## Architecture Details

### State Encoding (Fixed)

```python
from explainrl.models.network import StateEncoder

encoder = StateEncoder(max_board_size=16, max_tiles=10)
encoded = encoder.encode(game_state)
# Always returns: (31, 16, 16) tensor
# 31 = 1 + 3×10 channels
# 16 = max_board_size
```

### Network Forward Pass

```python
from explainrl.models import TilerSliderNet

model = TilerSliderNet(hidden_dim=256, num_recursive_steps=3)

# Encode state (31 channels)
encoded = model.encoder.encode(state)  # (1, 31, 16, 16)

# Extract features with conv layers
features = model.embedding(encoded)  # (1, 256, 16, 16)

# Flatten and add positional encoding
features = features.flatten(2).transpose(1, 2)  # (1, 256, 256)

# Apply TRM (3 iterations)
for _ in range(3):
    features = model.recursive_block(features)

# Pool and predict
pooled = model.pool(features.transpose(1, 2)).squeeze(-1)  # (1, 256)
value = model.value_head(pooled)  # (1, 1)
policy = model.policy_head(pooled)  # (1, 4)
```

## Training Results

The fix enables proper training. Example from a quick 10-iteration run:

```
Iteration 5/10
  Success Rate: 0.00%
  Avg Reward: -5.40
  Policy Loss: -0.3265
  Value Loss: 10.3295

Iteration 10/10
  Success Rate: 16.67%
  Avg Reward: -3.02
  Policy Loss: -0.2101
  Value Loss: 7.2552
```

The model learns to solve simple puzzles in just 10 iterations!

## Hyperparameter Tuning

### For Quick Experimentation
```bash
--hidden-dim 64
--num-recursive-steps 1
--trajectories-per-env 3
--num-epochs 2
--batch-size 16
```

### For Better Performance
```bash
--hidden-dim 256
--num-recursive-steps 3
--trajectories-per-env 10
--num-epochs 4
--batch-size 64
```

### For Maximum Performance
```bash
--hidden-dim 512
--num-recursive-steps 5
--num-heads 8
--trajectories-per-env 20
--num-epochs 6
--batch-size 128
```

## Monitoring Training

The trainer logs:
- **Success Rate**: % of puzzles solved
- **Avg Reward**: Average episode reward
- **Avg Length**: Average episode length
- **Policy Loss**: PPO policy loss
- **Value Loss**: Value function MSE loss
- **Entropy**: Policy entropy (exploration)

Checkpoints are saved to `--save-dir` every 100 iterations.

## Loading Trained Models

```python
import torch
from explainrl.models import TilerSliderNet

# Load checkpoint
checkpoint = torch.load('checkpoints/model_final.pt')

# Create model
model = TilerSliderNet(
    hidden_dim=256,
    num_recursive_steps=3
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])

# Use for inference
model.eval()
value, action_probs = model.predict(game_state)
```

## Common Issues

### Out of Memory
- Reduce `--hidden-dim`
- Reduce `--batch-size`
- Reduce `--trajectories-per-env`

### Training Too Slow
- Reduce `--num-recursive-steps`
- Use fewer puzzles initially
- Reduce `--num-epochs`

### Low Success Rate
- Increase training iterations
- Increase `--trajectories-per-env`
- Use simpler puzzles initially
- Increase `--entropy-coef` for more exploration

## Next Steps

1. **Experiment with hyperparameters** to find the best configuration
2. **Train on progressively harder puzzles** using curriculum learning
3. **Compare with BFS solutions** to validate learned policies
4. **Visualize learned policies** by running the model on test puzzles
5. **Combine with tree search** for even better performance

## Resources

- Model architecture: `explainrl/models/network.py`
- PPO training: `explainrl/models/ppo.py`
- Training script: `explainrl/models/train.py`
- Tests: `tests/test_models.py`
- Full documentation: `explainrl/models/README.md`
