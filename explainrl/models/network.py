"""
Neural network architecture for Tiler-Slider puzzle.

Uses Tiny Recursive Models (TRM) to enable deep reasoning about game states
through iterative refinement of representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

from explainrl.environment import GameState


class StateEncoder(nn.Module):
    """
    Encodes GameState into a multi-channel tensor representation.

    Channels:
    - Blocked cells (binary)
    - Current tile positions (one-hot per tile, or single channel for single-color)
    - Target positions (one-hot per tile, or single channel for single-color)
    - Distance to targets (for each tile)
    """

    def __init__(self, max_board_size: int = 16, max_tiles: int = 10):
        """
        Initialize state encoder.

        Args:
            max_board_size: Maximum board dimension to support
            max_tiles: Maximum number of tiles to support
        """
        super().__init__()
        self.max_board_size = max_board_size
        self.max_tiles = max_tiles

    def encode(self, state: GameState) -> torch.Tensor:
        """
        Encode game state to tensor representation.

        Always returns a fixed number of channels (1 + 3*max_tiles) for consistency,
        regardless of the actual number of tiles or multi_color setting.

        Args:
            state: GameState to encode

        Returns:
            Tensor of shape (max_channels, max_board_size, max_board_size)
            where max_channels = 1 + 3*max_tiles
        """
        size = state.size
        num_tiles = len(state.current_locations)

        # Always use max channels for consistent network input
        max_channels = 1 + 3 * self.max_tiles

        # Initialize tensor with zeros (unused channels stay zero)
        encoded = torch.zeros((max_channels, self.max_board_size, self.max_board_size))

        channel_idx = 0

        # Channel 0: Blocked cells
        for i in range(size):
            for j in range(size):
                if state.is_blocked[i, j]:
                    encoded[channel_idx, i, j] = 1.0
        channel_idx += 1

        # Channels 1 to max_tiles: Current positions (one channel per tile)
        # For single-color mode, we still use separate channels for consistency
        for tile_idx, (i, j) in enumerate(state.current_locations):
            if tile_idx < self.max_tiles:
                encoded[channel_idx + tile_idx, i, j] = 1.0
        channel_idx += self.max_tiles

        # Channels max_tiles+1 to 2*max_tiles: Target positions (one channel per tile)
        for tile_idx, (i, j) in enumerate(state.target_locations):
            if tile_idx < self.max_tiles:
                encoded[channel_idx + tile_idx, i, j] = 1.0
        channel_idx += self.max_tiles

        # Channels 2*max_tiles+1 to 3*max_tiles: Distance from each tile to target
        for tile_idx in range(min(num_tiles, self.max_tiles)):
            curr = state.current_locations[tile_idx]

            # Calculate distance based on mode
            if state.multi_color:
                # Multi-color: distance to specific target
                target = state.target_locations[tile_idx]
                distance = abs(curr[0] - target[0]) + abs(curr[1] - target[1])
            else:
                # Single-color: distance to nearest target
                distance = min(
                    abs(curr[0] - t[0]) + abs(curr[1] - t[1])
                    for t in state.target_locations
                )

            # Normalize distance by board size
            encoded[channel_idx + tile_idx, curr[0], curr[1]] = distance / (2 * size)
        channel_idx += self.max_tiles

        return encoded


class RecursiveBlock(nn.Module):
    """
    Single block of the Tiny Recursive Model.

    This block is applied multiple times to the same representation,
    allowing the model to "think" iteratively about the state.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        """
        Initialize recursive block.

        Args:
            hidden_dim: Dimension of hidden representations
            num_heads: Number of attention heads
        """
        super().__init__()

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply recursive block.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Output tensor of same shape
        """
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class TilerSliderNet(nn.Module):
    """
    Neural network for Tiler-Slider puzzle using Tiny Recursive Models.

    Architecture:
    1. State Encoding: Convert GameState to multi-channel tensor
    2. Embedding: Conv layers to extract features + flatten to sequence
    3. TRM: Apply RecursiveBlock multiple times for iterative reasoning
    4. Dual Heads:
       - Value Head: Estimate steps to completion
       - Policy Head: Probability distribution over actions
    """

    def __init__(
        self,
        max_board_size: int = 16,
        max_tiles: int = 10,
        hidden_dim: int = 256,
        num_recursive_steps: int = 3,
        num_heads: int = 4
    ):
        """
        Initialize network.

        Args:
            max_board_size: Maximum board size
            max_tiles: Maximum number of tiles
            hidden_dim: Hidden dimension for TRM
            num_recursive_steps: Number of times to apply recursive block
            num_recursive_heads: Number of attention heads in TRM
        """
        super().__init__()

        self.max_board_size = max_board_size
        self.max_tiles = max_tiles
        self.hidden_dim = hidden_dim
        self.num_recursive_steps = num_recursive_steps

        # State encoder
        self.encoder = StateEncoder(max_board_size, max_tiles)

        # Embedding network: Conv layers to process spatial info
        # Max channels: 1 + max_tiles*2 + max_tiles = 1 + 3*max_tiles
        max_channels = 1 + 3 * max_tiles

        self.embedding = nn.Sequential(
            nn.Conv2d(max_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Positional encoding for flattened spatial dimensions
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_board_size * max_board_size, hidden_dim)
        )

        # Tiny Recursive Model - single block applied multiple times
        self.recursive_block = RecursiveBlock(hidden_dim, num_heads)

        # Pooling to get single vector from sequence
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Value head: Predict number of steps to completion
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Policy head: Predict action probabilities
        # 4 actions: UP, DOWN, LEFT, RIGHT
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(
        self,
        state: GameState,
        return_embedding: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the network.

        Args:
            state: GameState to evaluate
            return_embedding: If True, also return the embedding

        Returns:
            Tuple of (value, policy_logits, embedding)
            - value: Estimated steps to completion (batch, 1)
            - policy_logits: Logits over 4 actions (batch, 4)
            - embedding: Hidden representation (batch, hidden_dim) if requested
        """
        # Encode state to tensor
        encoded = self.encoder.encode(state).unsqueeze(0)  # Add batch dim

        # Extract features with conv layers
        features = self.embedding(encoded)  # (batch, hidden_dim, H, W)

        # Flatten spatial dimensions to sequence
        batch_size = features.shape[0]
        features = features.flatten(2).transpose(1, 2)  # (batch, H*W, hidden_dim)

        # Add positional encoding
        seq_len = features.shape[1]
        features = features + self.pos_encoding[:, :seq_len, :]

        # Apply recursive block multiple times (TRM)
        x = features
        for _ in range(self.num_recursive_steps):
            x = self.recursive_block(x)

        # Pool to get single vector
        # x: (batch, seq_len, hidden_dim) -> (batch, hidden_dim, seq_len)
        x = x.transpose(1, 2)
        pooled = self.pool(x).squeeze(-1)  # (batch, hidden_dim)

        # Compute value and policy
        value = self.value_head(pooled)  # (batch, 1)
        policy_logits = self.policy_head(pooled)  # (batch, 4)

        if return_embedding:
            return value, policy_logits, pooled
        else:
            return value, policy_logits, None

    def predict(self, state: GameState) -> Tuple[float, torch.Tensor]:
        """
        Predict value and policy for a single state.

        Args:
            state: GameState to evaluate

        Returns:
            Tuple of (value, action_probs)
            - value: Estimated steps to completion
            - action_probs: Probability distribution over 4 actions
        """
        self.eval()
        with torch.no_grad():
            value, policy_logits, _ = self.forward(state)
            action_probs = F.softmax(policy_logits, dim=-1)
            return value.item(), action_probs.squeeze(0)

    def select_action(self, state: GameState, deterministic: bool = False) -> int:
        """
        Select an action given a state.

        Args:
            state: GameState to act on
            deterministic: If True, select argmax; else sample

        Returns:
            Action index (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        """
        _, action_probs = self.predict(state)

        if deterministic:
            return action_probs.argmax().item()
        else:
            return torch.multinomial(action_probs, 1).item()
