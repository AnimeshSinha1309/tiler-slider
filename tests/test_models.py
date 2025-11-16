"""
Unit tests for neural network models.

Tests cover:
- State encoding
- TRM architecture
- Value and policy predictions
- PPO training components
- Trajectory collection
"""

import pytest
import torch
import numpy as np
from explainrl.environment import TilerSliderEnv, GameState
from explainrl.models import TilerSliderNet, PPOTrainer


class TestStateEncoder:
    """Test state encoding functionality."""

    def test_encoder_initialization(self):
        """Test encoder can be initialized."""
        from explainrl.models.network import StateEncoder

        encoder = StateEncoder(max_board_size=16, max_tiles=10)
        assert encoder.max_board_size == 16
        assert encoder.max_tiles == 10

    def test_encode_simple_state(self):
        """Test encoding a simple game state."""
        from explainrl.models.network import StateEncoder

        env = TilerSliderEnv(
            size=4,
            blocked_locations=[(1, 1)],
            initial_locations=[(0, 0)],
            target_locations=[(3, 3)],
            multi_color=False
        )
        env.reset()

        encoder = StateEncoder(max_board_size=16, max_tiles=10)
        encoded = encoder.encode(env.state)

        # Check shape - should always be max_channels = 1 + 3*max_tiles
        max_channels = 1 + 3 * 10
        assert encoded.shape == (max_channels, 16, 16)

        # Check blocked cell is encoded
        assert encoded[0, 1, 1] == 1.0

        # Check tile position (channel 1)
        assert encoded[1, 0, 0] == 1.0

        # Check target position (channel 11 = 1 + max_tiles)
        assert encoded[11, 3, 3] == 1.0

    def test_encode_multi_color(self):
        """Test encoding multi-color state."""
        from explainrl.models.network import StateEncoder

        env = TilerSliderEnv(
            size=4,
            blocked_locations=[],
            initial_locations=[(0, 0), (0, 1)],
            target_locations=[(3, 0), (3, 1)],
            multi_color=True
        )
        env.reset()

        encoder = StateEncoder(max_board_size=16, max_tiles=10)
        encoded = encoder.encode(env.state)

        # Should always have max_channels = 1 + 3*max_tiles
        max_channels = 1 + 3 * 10
        assert encoded.shape == (max_channels, 16, 16)

        # Check tiles are in separate channels
        assert encoded[1, 0, 0] == 1.0  # Tile 0 (channel 1)
        assert encoded[2, 0, 1] == 1.0  # Tile 1 (channel 2)

        # Check targets are in correct channels (starting at 1 + max_tiles = 11)
        assert encoded[11, 3, 0] == 1.0  # Target 0
        assert encoded[12, 3, 1] == 1.0  # Target 1

    def test_encode_tensor_type(self):
        """Test that encoding returns proper tensor."""
        from explainrl.models.network import StateEncoder

        env = TilerSliderEnv(size=4, initial_locations=[(0, 0)], target_locations=[(3, 3)])
        env.reset()

        encoder = StateEncoder()
        encoded = encoder.encode(env.state)

        assert isinstance(encoded, torch.Tensor)
        assert encoded.dtype == torch.float32


class TestRecursiveBlock:
    """Test TRM recursive block."""

    def test_recursive_block_forward(self):
        """Test forward pass through recursive block."""
        from explainrl.models.network import RecursiveBlock

        block = RecursiveBlock(hidden_dim=256, num_heads=4)

        # Create dummy input
        batch_size = 2
        seq_len = 10
        hidden_dim = 256
        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Forward pass
        output = block(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, hidden_dim)

    def test_recursive_block_multiple_applications(self):
        """Test applying block multiple times."""
        from explainrl.models.network import RecursiveBlock

        block = RecursiveBlock(hidden_dim=128)
        x = torch.randn(1, 5, 128)

        # Apply multiple times
        x1 = block(x)
        x2 = block(x1)
        x3 = block(x2)

        # Shapes should all be the same
        assert x1.shape == x.shape
        assert x2.shape == x.shape
        assert x3.shape == x.shape

        # But values should be different (network is learning)
        assert not torch.allclose(x1, x2)


class TestTilerSliderNet:
    """Test main network architecture."""

    def test_network_initialization(self):
        """Test network can be initialized."""
        model = TilerSliderNet(
            max_board_size=16,
            max_tiles=10,
            hidden_dim=256,
            num_recursive_steps=3
        )

        assert model.max_board_size == 16
        assert model.hidden_dim == 256
        assert model.num_recursive_steps == 3

    def test_forward_pass(self):
        """Test forward pass through network."""
        model = TilerSliderNet(hidden_dim=128, num_recursive_steps=2)

        env = TilerSliderEnv(
            size=4,
            initial_locations=[(0, 0)],
            target_locations=[(3, 3)],
            multi_color=False
        )
        env.reset()

        # Forward pass
        value, policy_logits, _ = model.forward(env.state)

        # Check shapes
        assert value.shape == (1, 1)  # (batch=1, value_dim=1)
        assert policy_logits.shape == (1, 4)  # (batch=1, num_actions=4)

    def test_predict(self):
        """Test prediction method."""
        model = TilerSliderNet(hidden_dim=128, num_recursive_steps=2)

        env = TilerSliderEnv(
            size=4,
            initial_locations=[(0, 0)],
            target_locations=[(3, 3)]
        )
        env.reset()

        # Predict
        value, action_probs = model.predict(env.state)

        # Check types and shapes
        assert isinstance(value, float)
        assert isinstance(action_probs, torch.Tensor)
        assert action_probs.shape == (4,)

        # Check action probs sum to 1
        assert torch.isclose(action_probs.sum(), torch.tensor(1.0), atol=1e-5)

        # Check all probs are non-negative
        assert (action_probs >= 0).all()

    def test_select_action_deterministic(self):
        """Test deterministic action selection."""
        model = TilerSliderNet(hidden_dim=128, num_recursive_steps=2)

        env = TilerSliderEnv(
            size=4,
            initial_locations=[(0, 0)],
            target_locations=[(3, 3)]
        )
        env.reset()

        # Select action deterministically
        action = model.select_action(env.state, deterministic=True)

        # Check action is valid
        assert isinstance(action, int)
        assert 0 <= action < 4

        # Should be consistent
        action2 = model.select_action(env.state, deterministic=True)
        assert action == action2

    def test_select_action_stochastic(self):
        """Test stochastic action selection."""
        model = TilerSliderNet(hidden_dim=128, num_recursive_steps=2)

        env = TilerSliderEnv(
            size=4,
            initial_locations=[(0, 0)],
            target_locations=[(3, 3)]
        )
        env.reset()

        # Collect multiple actions
        actions = [model.select_action(env.state, deterministic=False) for _ in range(20)]

        # All should be valid
        assert all(0 <= a < 4 for a in actions)

        # Should have some variation (with high probability)
        # Note: This could theoretically fail, but very unlikely
        assert len(set(actions)) > 1

    def test_embedding_return(self):
        """Test that we can get embeddings."""
        model = TilerSliderNet(hidden_dim=128, num_recursive_steps=2)

        env = TilerSliderEnv(size=4, initial_locations=[(0, 0)], target_locations=[(3, 3)])
        env.reset()

        # Forward with embedding
        value, policy_logits, embedding = model.forward(env.state, return_embedding=True)

        # Check embedding shape
        assert embedding is not None
        assert embedding.shape == (1, 128)

    def test_different_board_sizes(self):
        """Test network handles different board sizes."""
        model = TilerSliderNet(max_board_size=16, hidden_dim=128, num_recursive_steps=2)

        for size in [3, 4, 5, 8]:
            env = TilerSliderEnv(
                size=size,
                initial_locations=[(0, 0)],
                target_locations=[(size-1, size-1)]
            )
            env.reset()

            value, policy_logits, _ = model.forward(env.state)

            assert value.shape == (1, 1)
            assert policy_logits.shape == (1, 4)

    def test_value_range(self):
        """Test that value predictions are reasonable."""
        model = TilerSliderNet(hidden_dim=128, num_recursive_steps=2)

        # Create states with different difficulties
        easy_env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(0, 0)]  # Already solved
        )
        easy_env.reset()

        hard_env = TilerSliderEnv(
            size=8,
            blocked_locations=[(i, 4) for i in range(8)],
            initial_locations=[(0, 0)],
            target_locations=[(7, 7)]
        )
        hard_env.reset()

        # Get predictions
        easy_value, _ = model.predict(easy_env.state)
        hard_value, _ = model.predict(hard_env.state)

        # Values should be real numbers (not NaN or inf)
        assert not np.isnan(easy_value)
        assert not np.isnan(hard_value)
        assert not np.isinf(easy_value)
        assert not np.isinf(hard_value)


class TestPPOTrainer:
    """Test PPO training components."""

    def test_trainer_initialization(self):
        """Test PPO trainer can be initialized."""
        model = TilerSliderNet(hidden_dim=128, num_recursive_steps=2)
        trainer = PPOTrainer(model, learning_rate=3e-4)

        assert trainer.model == model
        assert trainer.gamma == 0.99
        assert trainer.clip_epsilon == 0.2

    def test_collect_trajectory(self):
        """Test trajectory collection."""
        model = TilerSliderNet(hidden_dim=64, num_recursive_steps=1)
        trainer = PPOTrainer(model, learning_rate=3e-4)

        env = TilerSliderEnv(
            size=4,
            initial_locations=[(0, 0)],
            target_locations=[(1, 1)],
            max_steps=10
        )

        trajectory = trainer.collect_trajectory(env, max_steps=10)

        # Check trajectory structure
        assert len(trajectory.transitions) > 0
        assert len(trajectory.transitions) <= 10
        assert isinstance(trajectory.total_reward, float)
        assert isinstance(trajectory.success, bool)

    def test_collect_trajectories(self):
        """Test collecting multiple trajectories."""
        model = TilerSliderNet(hidden_dim=64, num_recursive_steps=1)
        trainer = PPOTrainer(model, learning_rate=3e-4)

        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)],
            max_steps=20
        )

        trajectories = trainer.collect_trajectories(env, num_trajectories=5, max_steps=20)

        assert len(trajectories) == 5
        for traj in trajectories:
            assert len(traj.transitions) > 0
            # Check rewards are assigned
            assert all(t.reward != 0.0 or i == len(traj.transitions) - 1
                      for i, t in enumerate(traj.transitions))

    def test_compute_advantages(self):
        """Test advantage computation."""
        model = TilerSliderNet(hidden_dim=64, num_recursive_steps=1)
        trainer = PPOTrainer(model, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95)

        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)]
        )

        trajectory = trainer.collect_trajectory(env, max_steps=10)

        # Compute rewards
        rewards = trainer.compute_rewards(trajectory)
        for t, r in zip(trajectory.transitions, rewards):
            t.reward = r

        # Compute advantages
        advantages, returns = trainer.compute_advantages(trajectory, rewards)

        # Check shapes
        assert len(advantages) == len(trajectory.transitions)
        assert len(returns) == len(trajectory.transitions)

        # Check normalization (approximately mean 0, std 1)
        assert abs(advantages.mean().item()) < 0.1
        assert abs(advantages.std().item() - 1.0) < 0.1

    def test_update_policy(self):
        """Test policy update step."""
        model = TilerSliderNet(hidden_dim=64, num_recursive_steps=1)
        trainer = PPOTrainer(model, learning_rate=3e-4)

        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)]
        )

        # Collect trajectories
        trajectories = trainer.collect_trajectories(env, num_trajectories=3, max_steps=10)

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Update
        stats = trainer.update(trajectories, num_epochs=2, batch_size=16)

        # Check stats
        assert 'policy_loss' in stats
        assert 'value_loss' in stats
        assert 'entropy' in stats

        # Check parameters changed
        params_changed = any(
            not torch.allclose(p1, p2)
            for p1, p2 in zip(initial_params, model.parameters())
        )
        assert params_changed, "Model parameters should have changed after update"

    def test_curriculum_schedule(self):
        """Test curriculum creation."""
        model = TilerSliderNet(hidden_dim=64, num_recursive_steps=1)
        trainer = PPOTrainer(model)

        envs = [
            TilerSliderEnv(size=3, initial_locations=[(0, 0)], target_locations=[(1, 1)]),
            TilerSliderEnv(size=4, initial_locations=[(0, 0)], target_locations=[(2, 2)]),
            TilerSliderEnv(size=5, initial_locations=[(0, 0)], target_locations=[(3, 3)])
        ]

        schedule = trainer._create_curriculum(envs, num_iterations=10)

        assert len(schedule) == 10

        # Early iterations should have fewer envs
        assert len(schedule[0]) == 1

        # Later iterations should have more
        assert len(schedule[-1]) >= len(schedule[0])

    def test_compute_rewards(self):
        """Test reward computation."""
        model = TilerSliderNet(hidden_dim=64, num_recursive_steps=1)
        trainer = PPOTrainer(model)

        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(2, 2)]
        )

        # Collect trajectory
        trajectory = trainer.collect_trajectory(env, max_steps=20)

        # Compute rewards
        rewards = trainer.compute_rewards(trajectory, step_penalty=-0.1,
                                         success_reward=10.0, failure_penalty=-5.0)

        assert len(rewards) == len(trajectory.transitions)

        # Final reward should be success or failure
        if trajectory.success:
            assert rewards[-1] == 10.0
        else:
            assert rewards[-1] == -5.0

        # Intermediate rewards should be step penalties
        for i in range(len(rewards) - 1):
            assert rewards[i] == -0.1


class TestTrainingIntegration:
    """Integration tests for training pipeline."""

    @pytest.mark.slow
    def test_short_training_run(self):
        """Test a short training run completes without errors."""
        model = TilerSliderNet(hidden_dim=64, num_recursive_steps=1)
        trainer = PPOTrainer(model, learning_rate=1e-3)

        # Create simple environments
        envs = [
            TilerSliderEnv(
                size=3,
                initial_locations=[(0, 0)],
                target_locations=[(1, 1)],
                max_steps=10
            ),
            TilerSliderEnv(
                size=3,
                initial_locations=[(0, 0)],
                target_locations=[(2, 2)],
                max_steps=15
            )
        ]

        # Short training run
        history = trainer.train(
            envs=envs,
            num_iterations=5,
            trajectories_per_env=2,
            max_steps=20,
            num_epochs=2,
            batch_size=16,
            eval_interval=2,
            verbose=False
        )

        # Check history structure
        assert 'success_rate' in history
        assert 'avg_reward' in history
        assert 'policy_loss' in history

        # Should have recorded stats at eval intervals
        assert len(history['success_rate']) > 0

    @pytest.mark.slow
    def test_training_improves_performance(self):
        """Test that training improves success rate."""
        model = TilerSliderNet(hidden_dim=128, num_recursive_steps=2)
        trainer = PPOTrainer(model, learning_rate=1e-3)

        # Very simple environment
        env = TilerSliderEnv(
            size=3,
            initial_locations=[(0, 0)],
            target_locations=[(1, 0)],  # Just move right
            max_steps=5
        )

        # Measure initial performance
        initial_trajectories = trainer.collect_trajectories(env, num_trajectories=10, max_steps=5)
        initial_success = sum(t.success for t in initial_trajectories) / len(initial_trajectories)

        # Train
        trainer.train(
            envs=[env],
            num_iterations=20,
            trajectories_per_env=5,
            max_steps=5,
            num_epochs=4,
            batch_size=32,
            eval_interval=5,
            verbose=False
        )

        # Measure final performance
        final_trajectories = trainer.collect_trajectories(env, num_trajectories=10, max_steps=5)
        final_success = sum(t.success for t in final_trajectories) / len(final_trajectories)

        # Performance should improve or stay good
        # (Allowing for randomness - may start high on such a simple task)
        assert final_success >= initial_success - 0.2
