#!/usr/bin/env python3
"""
Training script for TilerSliderNet using PPO.

Trains a neural network to solve tiler-slider puzzles using
Proximal Policy Optimization with curriculum learning.
"""

import argparse
import os
import sys
from typing import List

import torch

from explainrl.environment import TilerSliderEnv, ImageLoader
from explainrl.models import TilerSliderNet, PPOTrainer


def load_puzzles(
    num_puzzles: int = 20,
    multi_color: bool = True,
    data_dir: str = 'data'
) -> List[TilerSliderEnv]:
    """
    Load puzzles for training.

    Args:
        num_puzzles: Number of puzzles to load
        multi_color: Load multi-color puzzles if True
        data_dir: Directory containing puzzle images

    Returns:
        List of TilerSliderEnv instances, sorted by difficulty
    """
    # Save current directory
    original_dir = os.getcwd()

    try:
        # Change to data directory
        os.chdir(data_dir)

        # Load puzzle images
        loader = ImageLoader()

        puzzles = []
        for idx, item in enumerate(loader):
            # Filter by type
            is_multi = 'multi' in item.name.lower()
            if is_multi != multi_color:
                continue

            # Parse puzzle
            level = ImageLoader.parse_puzzle_image(item.puzzle_image, multiple_colors=multi_color)

            # Create environment
            env = TilerSliderEnv.from_level(level, max_steps=100)
            puzzles.append((idx, env, level.size, len(level.initial_locations)))

            if len(puzzles) >= num_puzzles:
                break

        # Sort by difficulty (board size, then number of tiles)
        puzzles.sort(key=lambda x: (x[2], x[3]))

        return [env for _, env, _, _ in puzzles]

    finally:
        # Restore original directory
        os.chdir(original_dir)


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description='Train TilerSliderNet using PPO')

    # Model architecture
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension for TRM')
    parser.add_argument('--num-recursive-steps', type=int, default=3,
                       help='Number of TRM iterations')
    parser.add_argument('--num-heads', type=int, default=4,
                       help='Number of attention heads')

    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--clip-epsilon', type=float, default=0.2,
                       help='PPO clip epsilon')
    parser.add_argument('--value-coef', type=float, default=0.5,
                       help='Value loss coefficient')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                       help='Entropy bonus coefficient')

    # Training configuration
    parser.add_argument('--num-iterations', type=int, default=1000,
                       help='Number of training iterations')
    parser.add_argument('--trajectories-per-env', type=int, default=10,
                       help='Trajectories per environment per iteration')
    parser.add_argument('--max-steps', type=int, default=100,
                       help='Maximum steps per trajectory')
    parser.add_argument('--num-epochs', type=int, default=4,
                       help='Number of PPO epochs per update')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--eval-interval', type=int, default=10,
                       help='Evaluate every N iterations')

    # Puzzle configuration
    parser.add_argument('--num-puzzles', type=int, default=20,
                       help='Number of puzzles to use for training')
    parser.add_argument('--multi-color', action='store_true',
                       help='Train on multi-color puzzles')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing puzzle images')

    # Checkpointing
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--load-checkpoint', type=str,
                       help='Path to checkpoint to resume from')

    # Device
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to train on')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load puzzles
    print(f"Loading {args.num_puzzles} puzzles...")
    puzzles = load_puzzles(
        num_puzzles=args.num_puzzles,
        multi_color=args.multi_color,
        data_dir=args.data_dir
    )
    print(f"Loaded {len(puzzles)} puzzles")

    if len(puzzles) == 0:
        print("No puzzles loaded! Check data directory and puzzle type.")
        return 1

    # Create model
    print("Creating model...")
    model = TilerSliderNet(
        max_board_size=16,
        max_tiles=10,
        hidden_dim=args.hidden_dim,
        num_recursive_steps=args.num_recursive_steps,
        num_heads=args.num_heads
    )

    # Create trainer
    print("Creating trainer...")
    trainer = PPOTrainer(
        model=model,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        device=args.device
    )

    # Load checkpoint if specified
    start_iteration = 0
    if args.load_checkpoint:
        print(f"Loading checkpoint from {args.load_checkpoint}...")
        start_iteration = trainer.load_checkpoint(args.load_checkpoint)

    # Train
    print("Starting training...")
    print("-" * 60)

    save_path = os.path.join(args.save_dir, 'model')

    history = trainer.train(
        envs=puzzles,
        num_iterations=args.num_iterations,
        trajectories_per_env=args.trajectories_per_env,
        max_steps=args.max_steps,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        save_path=save_path,
        verbose=True
    )

    print("\nTraining complete!")
    print(f"Final success rate: {history['success_rate'][-1]:.2%}")
    print(f"Final avg reward: {history['avg_reward'][-1]:.2f}")

    # Save final model
    final_path = os.path.join(args.save_dir, 'model_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'args': vars(args)
    }, final_path)
    print(f"Saved final model to {final_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
