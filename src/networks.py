"""
Neural Network architectures for MADDPG.

Follows sumo-marl reference implementation:
- Actor: obs -> 64 -> 64 -> action_probs (with LeakyReLU, softmax output)
- Critic: (global_state + all_actions) -> 64 -> 64 -> Q-value (with LeakyReLU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def soft_update(source: nn.Module, target: nn.Module, tau: float):
    """
    Soft update target network parameters.

    target = tau * source + (1 - tau) * target

    Args:
        source: Source network
        target: Target network
        tau: Interpolation parameter (0 < tau << 1)
    """
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


def hard_update(source: nn.Module, target: nn.Module):
    """
    Hard update: copy all parameters from source to target.

    Args:
        source: Source network
        target: Target network
    """
    target.load_state_dict(source.state_dict())


class Actor(nn.Module):
    """
    Actor network for individual agent (matches sumo-marl).

    Maps local observation to action probabilities via softmax.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: List[int] = None,
    ):
        """
        Initialize Actor network.

        Args:
            obs_dim: Dimension of local observation
            action_dim: Number of actions
            hidden: List of hidden layer sizes (default: [64, 64])
        """
        super(Actor, self).__init__()

        if hidden is None:
            hidden = [64, 64]

        self.layers = nn.ModuleList()
        input_dims = [obs_dim] + hidden
        output_dims = hidden + [action_dim]

        for in_dim, out_dim in zip(input_dims[:-1], output_dims[:-1]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.LeakyReLU())

        # Final layer (no activation - softmax applied in forward)
        self.layers.append(nn.Linear(input_dims[-1], output_dims[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns action probabilities.

        Args:
            x: Local observation tensor of shape (batch, obs_dim) or (obs_dim,)

        Returns:
            Action probabilities of shape (batch, action_dim) or (action_dim,)
        """
        for layer in self.layers:
            x = layer(x)
        return F.softmax(x, dim=-1)


class Critic(nn.Module):
    """
    Centralized Critic network for MADDPG (matches sumo-marl).

    Maps global state (all observations) and all actions to Q-value.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: List[int] = None,
    ):
        """
        Initialize Critic network.

        Args:
            state_dim: Total dimension of all agents' observations (n_agents * obs_dim)
            action_dim: Total dimension of all agents' actions (n_agents * action_dim)
            hidden: List of hidden layer sizes (default: [64, 64])
        """
        super(Critic, self).__init__()

        if hidden is None:
            hidden = [64, 64]

        self.layers = nn.ModuleList()
        input_dim = state_dim + action_dim

        for h in hidden:
            self.layers.append(nn.Linear(input_dim, h))
            self.layers.append(nn.LeakyReLU())
            input_dim = h

        # Output: single Q-value
        self.layers.append(nn.Linear(input_dim, 1))

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Global state tensor of shape (batch, state_dim)
            action: All actions tensor of shape (batch, action_dim)

        Returns:
            Q-value of shape (batch, 1)
        """
        x = torch.cat([state, action], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    # Test networks
    batch_size = 32
    n_agents = 4
    obs_dim = 10
    action_dim = 2

    # Test Actor
    print("Testing Actor...")
    actor = Actor(obs_dim, action_dim)
    obs = torch.randn(batch_size, obs_dim)
    action_probs = actor(obs)
    print(f"Actor output shape: {action_probs.shape}")
    print(f"Action probs sum: {action_probs.sum(dim=-1)[:5]}")  # Should be ~1.0

    # Test single observation
    single_obs = torch.randn(obs_dim)
    single_probs = actor(single_obs)
    print(f"Single obs output shape: {single_probs.shape}")

    # Test Critic
    print("\nTesting Critic...")
    state_dim = n_agents * obs_dim
    total_action_dim = n_agents * action_dim
    critic = Critic(state_dim, total_action_dim)
    global_state = torch.randn(batch_size, state_dim)
    global_actions = torch.randn(batch_size, total_action_dim)
    q_value = critic(global_state, global_actions)
    print(f"Critic output shape: {q_value.shape}")

    # Test soft update
    print("\nTesting soft update...")
    target_actor = Actor(obs_dim, action_dim)
    hard_update(actor, target_actor)
    soft_update(actor, target_actor, tau=0.001)
    print("Soft update successful!")

    print("\nNetwork tests passed!")
