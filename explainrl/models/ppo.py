"""
Proximal Policy Optimization (PPO) training for Tiler-Slider puzzle.

Implements:
- Trajectory collection using current policy
- Advantage estimation with GAE
- PPO loss with clipped objective
- Curriculum learning from easy to hard puzzles
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass
import os
from tqdm import tqdm

from explainrl.environment import TilerSliderEnv, GameState, ImageLoader
from explainrl.models.network import TilerSliderNet
from explainrl.models.device_utils import get_device


@dataclass
class Transition:
    """Single transition in a trajectory."""
    state: GameState
    action: int
    reward: float
    next_state: GameState
    done: bool
    value: float
    log_prob: float


@dataclass
class Trajectory:
    """Complete trajectory from start to finish."""
    transitions: List[Transition]
    total_reward: float
    length: int
    success: bool


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for Tiler-Slider puzzle.

    Features:
    - Collects trajectories using current policy
    - Computes advantages using GAE (Generalized Advantage Estimation)
    - Optimizes policy using PPO clipped objective
    - Supports curriculum learning
    """

    def __init__(
        self,
        model: TilerSliderNet,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = None
    ):
        """
        Initialize PPO trainer.

        Args:
            model: TilerSliderNet to train
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: Lambda for GAE
            clip_epsilon: Clipping parameter for PPO
            value_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to train on (None = auto-detect, 'cpu', 'cuda', or 'mps')
        """
        # Auto-detect device if not specified
        if device is None:
            device = get_device(prefer_gpu=True)

        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Training statistics
        self.stats = {
            'total_episodes': 0,
            'total_steps': 0,
            'success_rate': 0.0,
            'avg_reward': 0.0,
            'avg_length': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0
        }

    def compute_rewards(
        self,
        trajectory: Trajectory,
        step_penalty: float = -0.1,
        success_reward: float = 10.0,
        failure_penalty: float = -5.0
    ) -> List[float]:
        """
        Compute rewards for trajectory.

        Args:
            trajectory: Trajectory to compute rewards for
            step_penalty: Penalty per step
            success_reward: Reward for solving puzzle
            failure_penalty: Penalty for failing to solve

        Returns:
            List of rewards for each transition
        """
        rewards = []
        for i, transition in enumerate(trajectory.transitions):
            if transition.done:
                # Last step: give success or failure reward
                if trajectory.success:
                    reward = success_reward
                else:
                    reward = failure_penalty
            else:
                # Regular step: small penalty to encourage efficiency
                reward = step_penalty

            rewards.append(reward)

        return rewards

    def compute_advantages(
        self,
        trajectory: Trajectory,
        rewards: List[float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using GAE.

        Args:
            trajectory: Trajectory to compute advantages for
            rewards: Rewards for each transition

        Returns:
            Tuple of (advantages, returns)
        """
        values = torch.tensor([t.value for t in trajectory.transitions])
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        # Compute TD errors
        next_values = torch.zeros_like(values)
        next_values[:-1] = values[1:]
        # Last state value is 0 if done, otherwise use current estimate
        if trajectory.transitions[-1].done:
            next_values[-1] = 0.0

        td_errors = rewards_tensor + self.gamma * next_values - values

        # Compute GAE
        advantages = torch.zeros_like(values)
        gae = 0.0
        for t in reversed(range(len(trajectory.transitions))):
            gae = td_errors[t] + self.gamma * self.gae_lambda * gae
            advantages[t] = gae

        # Compute returns (for value function training)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def collect_trajectory(
        self,
        env: TilerSliderEnv,
        max_steps: int = 100,
        deterministic: bool = False
    ) -> Trajectory:
        """
        Collect a single trajectory using current policy.

        Args:
            env: Environment to collect from
            max_steps: Maximum steps before termination
            deterministic: Use deterministic policy if True

        Returns:
            Collected trajectory
        """
        self.model.eval()

        transitions = []
        obs = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            # Get action from policy
            with torch.no_grad():
                value, policy_logits, _ = self.model.forward(env.state)
                action_probs = F.softmax(policy_logits, dim=-1).squeeze(0)

                if deterministic:
                    action = action_probs.argmax().item()
                else:
                    action = torch.multinomial(action_probs, 1).item()

                log_prob = torch.log(action_probs[action] + 1e-8).item()

            # Take action
            move = GameState.Move(action)
            next_obs, done, info = env.step(move)

            # Store transition (reward will be computed later)
            transition = Transition(
                state=env.state.copy() if not done else None,
                action=action,
                reward=0.0,  # Will be filled later
                next_state=env.state.copy() if not done else None,
                done=done,
                value=value.item(),
                log_prob=log_prob
            )
            transitions.append(transition)

            if done:
                break

        success = env.state.is_won() if env.state else False

        trajectory = Trajectory(
            transitions=transitions,
            total_reward=0.0,  # Will be computed later
            length=len(transitions),
            success=success
        )

        return trajectory

    def collect_trajectories(
        self,
        env: TilerSliderEnv,
        num_trajectories: int,
        max_steps: int = 100
    ) -> List[Trajectory]:
        """
        Collect multiple trajectories.

        Args:
            env: Environment to collect from
            num_trajectories: Number of trajectories to collect
            max_steps: Maximum steps per trajectory

        Returns:
            List of collected trajectories
        """
        trajectories = []

        for _ in range(num_trajectories):
            trajectory = self.collect_trajectory(env, max_steps)

            # Compute rewards
            rewards = self.compute_rewards(trajectory)
            for transition, reward in zip(trajectory.transitions, rewards):
                transition.reward = reward

            trajectory.total_reward = sum(rewards)
            trajectories.append(trajectory)

        return trajectories

    def update(
        self,
        trajectories: List[Trajectory],
        num_epochs: int = 4,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Update policy using PPO.

        Args:
            trajectories: List of trajectories to train on
            num_epochs: Number of epochs to train
            batch_size: Batch size for training

        Returns:
            Dictionary of training statistics
        """
        self.model.train()

        # Prepare data
        states = []
        actions = []
        old_log_probs = []
        advantages = []
        returns = []

        for trajectory in trajectories:
            # Compute advantages
            rewards = [t.reward for t in trajectory.transitions]
            traj_advantages, traj_returns = self.compute_advantages(trajectory, rewards)

            # Collect data
            for i, transition in enumerate(trajectory.transitions):
                # Skip the last state if done (no state to encode)
                if transition.state is not None:
                    states.append(transition.state)
                    actions.append(transition.action)
                    old_log_probs.append(transition.log_prob)
                    advantages.append(traj_advantages[i].item())
                    returns.append(traj_returns[i].item())

        # Convert to tensors
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Training loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for epoch in range(num_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))

            for start_idx in range(0, len(states), batch_size):
                end_idx = min(start_idx + batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]

                # Get batch
                batch_states = [states[i] for i in batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Forward pass for each state in batch
                batch_values = []
                batch_policy_logits = []

                for state in batch_states:
                    value, policy_logits, _ = self.model.forward(state)
                    batch_values.append(value)
                    batch_policy_logits.append(policy_logits)

                batch_values = torch.cat(batch_values, dim=0)
                batch_policy_logits = torch.cat(batch_policy_logits, dim=0)

                # Compute new log probs and entropy
                action_probs = F.softmax(batch_policy_logits, dim=-1)
                action_log_probs = F.log_softmax(batch_policy_logits, dim=-1)
                batch_new_log_probs = action_log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

                entropy = -(action_probs * action_log_probs).sum(dim=-1).mean()

                # Compute PPO loss
                ratio = torch.exp(batch_new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                value_loss = F.mse_loss(batch_values.squeeze(1), batch_returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        # Return average statistics
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }

    def train(
        self,
        envs: List[TilerSliderEnv],
        num_iterations: int = 1000,
        trajectories_per_env: int = 10,
        max_steps: int = 100,
        num_epochs: int = 4,
        batch_size: int = 64,
        eval_interval: int = 10,
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model using PPO with curriculum learning.

        Args:
            envs: List of environments (ordered easy to hard for curriculum)
            num_iterations: Number of training iterations
            trajectories_per_env: Number of trajectories per environment
            max_steps: Maximum steps per trajectory
            num_epochs: Number of PPO epochs per update
            batch_size: Batch size for training
            eval_interval: Evaluate every N iterations
            save_path: Path to save model checkpoints
            verbose: Print progress if True

        Returns:
            Dictionary of training curves
        """
        history = {
            'success_rate': [],
            'avg_reward': [],
            'avg_length': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': []
        }

        # Curriculum: start with easier puzzles
        curriculum_schedule = self._create_curriculum(envs, num_iterations)

        if verbose:
            print(f"Starting PPO training for {num_iterations} iterations")
            print(f"Curriculum: {len(envs)} environments")
            print(f"Trajectories per env: {trajectories_per_env}")
            print("-" * 60)

        for iteration in tqdm(range(num_iterations), desc="Training", disable=not verbose):
            # Get environments for this iteration based on curriculum
            active_envs = curriculum_schedule[iteration]

            # Collect trajectories
            all_trajectories = []
            for env in active_envs:
                trajectories = self.collect_trajectories(env, trajectories_per_env, max_steps)
                all_trajectories.extend(trajectories)

            # Update policy
            update_stats = self.update(all_trajectories, num_epochs, batch_size)

            # Update statistics
            success_rate = sum(t.success for t in all_trajectories) / len(all_trajectories)
            avg_reward = np.mean([t.total_reward for t in all_trajectories])
            avg_length = np.mean([t.length for t in all_trajectories])

            self.stats['total_episodes'] += len(all_trajectories)
            self.stats['total_steps'] += sum(t.length for t in all_trajectories)
            self.stats['success_rate'] = success_rate
            self.stats['avg_reward'] = avg_reward
            self.stats['avg_length'] = avg_length
            self.stats['policy_loss'] = update_stats['policy_loss']
            self.stats['value_loss'] = update_stats['value_loss']
            self.stats['entropy'] = update_stats['entropy']

            # Log and save
            if (iteration + 1) % eval_interval == 0:
                history['success_rate'].append(success_rate)
                history['avg_reward'].append(avg_reward)
                history['avg_length'].append(avg_length)
                history['policy_loss'].append(update_stats['policy_loss'])
                history['value_loss'].append(update_stats['value_loss'])
                history['entropy'].append(update_stats['entropy'])

                if verbose:
                    print(f"\nIteration {iteration + 1}/{num_iterations}")
                    print(f"  Success Rate: {success_rate:.2%}")
                    print(f"  Avg Reward: {avg_reward:.2f}")
                    print(f"  Avg Length: {avg_length:.1f}")
                    print(f"  Policy Loss: {update_stats['policy_loss']:.4f}")
                    print(f"  Value Loss: {update_stats['value_loss']:.4f}")
                    print(f"  Entropy: {update_stats['entropy']:.4f}")

                if save_path and (iteration + 1) % (eval_interval * 10) == 0:
                    self.save_checkpoint(save_path, iteration + 1)

        if save_path:
            self.save_checkpoint(save_path, num_iterations)

        return history

    def _create_curriculum(
        self,
        envs: List[TilerSliderEnv],
        num_iterations: int
    ) -> List[List[TilerSliderEnv]]:
        """
        Create curriculum schedule.

        Gradually introduces harder puzzles over training.

        Args:
            envs: List of environments (ordered easy to hard)
            num_iterations: Total number of iterations

        Returns:
            List of environment lists for each iteration
        """
        schedule = []
        num_envs = len(envs)

        for iteration in range(num_iterations):
            # Gradually add more environments
            progress = iteration / num_iterations
            num_active = max(1, int(progress * num_envs) + 1)
            num_active = min(num_active, num_envs)

            # Use environments from easiest to current difficulty
            active_envs = envs[:num_active]
            schedule.append(active_envs)

        return schedule

    def save_checkpoint(self, path: str, iteration: int):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': self.stats
        }
        save_file = f"{path}_iter{iteration}.pt"
        torch.save(checkpoint, save_file)
        print(f"Saved checkpoint to {save_file}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.stats = checkpoint['stats']
        print(f"Loaded checkpoint from {path}")
        return checkpoint['iteration']
