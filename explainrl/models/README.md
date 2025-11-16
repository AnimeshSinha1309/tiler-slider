# Tiler-Slider Neural Network Models

This module implements neural network models for solving tiler-slider puzzles using reinforcement learning.

## Architecture

### TilerSliderNet

A neural network that uses **Tiny Recursive Models (TRM)** to enable deep reasoning about game states.

**Key Components:**
1. **State Encoder**: Converts `GameState` to multi-channel tensor representation
   - Blocked cells (binary)
   - Current tile positions (one-hot or aggregated)
   - Target positions (one-hot or aggregated)
   - Distance features (Manhattan distance to targets)

2. **Embedding Network**: Convolutional layers to extract spatial features
   - 3 conv layers with ReLU activations
   - Converts spatial grid to hidden dimension

3. **Tiny Recursive Model (TRM)**: Iterative refinement
   - Single RecursiveBlock (transformer-like) applied multiple times
   - Multi-head self-attention + feed-forward network
   - Enables "thinking" by processing the same representation iteratively

4. **Dual Heads**:
   - **Value Head**: Predicts number of steps to completion
   - **Policy Head**: Outputs probability distribution over 4 actions (UP, DOWN, LEFT, RIGHT)

**Parameters:**
- `max_board_size`: Maximum board dimension (default: 16)
- `max_tiles`: Maximum number of tiles (default: 10)
- `hidden_dim`: Hidden dimension for TRM (default: 256)
- `num_recursive_steps`: Number of TRM iterations (default: 3)
- `num_heads`: Number of attention heads (default: 4)

### Example Usage

```python
from explainrl.environment import TilerSliderEnv
from explainrl.models import TilerSliderNet

# Create model
model = TilerSliderNet(
    hidden_dim=256,
    num_recursive_steps=3,
    num_heads=4
)

# Create environment
env = TilerSliderEnv(
    size=4,
    initial_locations=[(0, 0)],
    target_locations=[(3, 3)]
)
env.reset()

# Get predictions
value, action_probs = model.predict(env.state)
print(f"Estimated steps to completion: {value}")
print(f"Action probabilities: {action_probs}")

# Select action
action = model.select_action(env.state, deterministic=False)
move = GameState.Move(action)
env.step(move)
```

## Training with PPO

### PPOTrainer

Implements **Proximal Policy Optimization** with curriculum learning.

**Features:**
- Trajectory collection using current policy
- Generalized Advantage Estimation (GAE)
- PPO clipped objective
- Curriculum learning: progressively introduces harder puzzles

**Hyperparameters:**
- `learning_rate`: Adam learning rate (default: 3e-4)
- `gamma`: Discount factor (default: 0.99)
- `gae_lambda`: GAE lambda (default: 0.95)
- `clip_epsilon`: PPO clipping parameter (default: 0.2)
- `value_coef`: Value loss coefficient (default: 0.5)
- `entropy_coef`: Entropy bonus (default: 0.01)

### Training Script

Train a model using the provided training script:

```bash
python -m explainrl.models.train \
    --num-puzzles 20 \
    --multi-color \
    --num-iterations 1000 \
    --hidden-dim 256 \
    --num-recursive-steps 3 \
    --learning-rate 3e-4 \
    --save-dir checkpoints/
```

**Options:**
- `--num-puzzles`: Number of puzzles for training (default: 20)
- `--multi-color`: Train on multi-color puzzles
- `--num-iterations`: Training iterations (default: 1000)
- `--hidden-dim`: Hidden dimension (default: 256)
- `--num-recursive-steps`: TRM iterations (default: 3)
- `--learning-rate`: Learning rate (default: 3e-4)
- `--save-dir`: Checkpoint directory (default: checkpoints/)
- `--device`: Device (cpu/cuda, default: cpu)

### Manual Training

```python
from explainrl.models import TilerSliderNet, PPOTrainer
from explainrl.environment import TilerSliderEnv

# Create model
model = TilerSliderNet(hidden_dim=256, num_recursive_steps=3)

# Create trainer
trainer = PPOTrainer(
    model=model,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95
)

# Create environments (easy to hard)
envs = [
    TilerSliderEnv(size=3, initial_locations=[(0, 0)], target_locations=[(1, 1)]),
    TilerSliderEnv(size=4, initial_locations=[(0, 0)], target_locations=[(2, 2)]),
    TilerSliderEnv(size=5, initial_locations=[(0, 0)], target_locations=[(3, 3)])
]

# Train
history = trainer.train(
    envs=envs,
    num_iterations=1000,
    trajectories_per_env=10,
    max_steps=100,
    num_epochs=4,
    batch_size=64,
    eval_interval=10,
    save_path='checkpoints/model',
    verbose=True
)

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history['success_rate'])
plt.title('Success Rate')
plt.xlabel('Evaluation')
plt.ylabel('Success Rate')

plt.subplot(1, 3, 2)
plt.plot(history['avg_reward'])
plt.title('Average Reward')
plt.xlabel('Evaluation')
plt.ylabel('Reward')

plt.subplot(1, 3, 3)
plt.plot(history['policy_loss'], label='Policy')
plt.plot(history['value_loss'], label='Value')
plt.plot(history['entropy'], label='Entropy')
plt.title('Losses')
plt.xlabel('Evaluation')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
```

## Curriculum Learning

The trainer automatically creates a curriculum schedule:
- **Early iterations**: Only easiest puzzles
- **Middle iterations**: Gradually add harder puzzles
- **Late iterations**: All puzzles included

This helps the model learn fundamental skills on simple problems before tackling complex ones.

## Testing

Run the test suite:

```bash
# All model tests
pytest tests/test_models.py -v

# Specific test classes
pytest tests/test_models.py::TestTilerSliderNet -v
pytest tests/test_models.py::TestPPOTrainer -v

# Skip slow tests
pytest tests/test_models.py -v -m "not slow"
```

## Model Architecture Details

### State Encoding

The encoder creates a multi-channel representation:

**Single-color mode (3 channels):**
1. Blocked cells (binary)
2. All current tile positions (binary)
3. All target positions (binary)
4. Minimum distance to any target (normalized)

**Multi-color mode (1 + 3×num_tiles channels):**
1. Blocked cells (binary)
2-N. Each tile's position (one-hot, N = num_tiles)
N+1 to 2N. Each target's position (one-hot)
2N+1 to 3N. Distance from each tile to its target (normalized)

### Tiny Recursive Model (TRM)

The TRM applies the same RecursiveBlock multiple times:

```
Input → Embed → TRM → TRM → TRM → Pool → [Value, Policy]
                 ↑      ↑      ↑
              Same block applied iteratively
```

Each application allows the model to refine its understanding of the state, similar to "thinking" steps.

### RecursiveBlock

A transformer-like block with:
1. Multi-head self-attention (4-8 heads)
2. Feed-forward network (4×expansion)
3. Layer normalization
4. Residual connections
5. Dropout (0.1)

## Requirements

- PyTorch >= 2.0
- NumPy >= 1.20
- tqdm (for progress bars)

Install with:
```bash
pip install torch numpy tqdm
```

## References

- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **GAE**: Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016)
- **Curriculum Learning**: Bengio et al., "Curriculum Learning" (2009)
- **Transformers**: Vaswani et al., "Attention Is All You Need" (2017)
