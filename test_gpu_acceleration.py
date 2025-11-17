#!/usr/bin/env python3
"""Test GPU acceleration with device auto-detection."""

import time
import torch
from explainrl.environment import TilerSliderEnv
from explainrl.models import TilerSliderNet, PPOTrainer, print_device_info

# Print device information
print_device_info()
print()

# Create a simple environment
env = TilerSliderEnv(
    size=4,
    initial_locations=[(0, 0), (0, 1)],
    target_locations=[(3, 0), (3, 1)],
    multi_color=True,
    max_steps=10
)

print("Testing GPU Acceleration")
print("=" * 60)

# Test 1: Device auto-detection
print("\n1. Testing device auto-detection...")
model = TilerSliderNet(hidden_dim=128, num_recursive_steps=2)
trainer = PPOTrainer(model, device=None)  # None = auto-detect
print(f"   Auto-detected device: {trainer.device}")
print(f"   Model device: {next(model.parameters()).device}")

# Test 2: Forward pass timing
print("\n2. Testing forward pass speed...")
env.reset()

# Warm-up
for _ in range(5):
    value, policy, _ = model.forward(env.state)

# Timed runs
num_runs = 100
start_time = time.time()
for _ in range(num_runs):
    value, policy, _ = model.forward(env.state)
    if trainer.device == 'cuda':
        torch.cuda.synchronize()  # Wait for GPU
elapsed = time.time() - start_time

print(f"   {num_runs} forward passes in {elapsed:.3f}s")
print(f"   Average: {elapsed/num_runs*1000:.2f}ms per pass")
print(f"   Throughput: {num_runs/elapsed:.0f} passes/sec")

# Test 3: Trajectory collection timing
print("\n3. Testing trajectory collection...")
start_time = time.time()
trajectories = trainer.collect_trajectories(env, num_trajectories=10, max_steps=10)
elapsed = time.time() - start_time

print(f"   Collected 10 trajectories in {elapsed:.3f}s")
print(f"   Average: {elapsed/10*1000:.0f}ms per trajectory")

# Test 4: Training update timing
print("\n4. Testing training update...")
start_time = time.time()
stats = trainer.update(trajectories, num_epochs=2, batch_size=16)
elapsed = time.time() - start_time

print(f"   Training update in {elapsed:.3f}s")
print(f"   Policy loss: {stats['policy_loss']:.4f}")
print(f"   Value loss: {stats['value_loss']:.4f}")

print("\n" + "=" * 60)
print("✓ GPU acceleration test completed!")
print(f"Using device: {trainer.device}")

if trainer.device == 'cuda':
    print(f"\nCUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
    print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e6:.1f} MB")
elif trainer.device == 'mps':
    print("\nUsing Apple Metal Performance Shaders (MPS)")
else:
    print("\nUsing CPU (no GPU acceleration)")
