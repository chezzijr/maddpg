"""
Experience Replay Buffer for MADDPG.

Stores transitions and provides efficient batch sampling.
"""

import random
from collections import deque
from typing import Tuple, Optional

import numpy as np
import torch


class ReplayBuffer:
    """
    Experience replay buffer for multi-agent transitions.

    Stores (states, actions, rewards, next_states, dones) tuples
    where states/actions are for ALL agents.
    """

    def __init__(
        self,
        capacity: int = 100000,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            device: Torch device for tensors (defaults to CPU)
        """
        self.capacity = capacity
        self.device = device or torch.device("cpu")
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def push(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        done: bool,
    ):
        """
        Store a transition.

        Args:
            states: All agents' observations, shape (n_agents, obs_dim)
            actions: All agents' action probabilities, shape (n_agents, action_dim)
            rewards: All agents' rewards, shape (n_agents,)
            next_states: All agents' next observations, shape (n_agents, obs_dim)
            done: Whether episode is done
        """
        # Flatten for storage
        transition = (
            states.flatten(),
            actions.flatten(),
            rewards,
            next_states.flatten(),
            float(done),
        )
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors
        """
        transitions = random.sample(self.buffer, batch_size)

        # Unzip transitions
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Convert to tensors
        states = torch.tensor(
            np.array(states), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            np.array(actions), dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor(
            np.array(rewards), dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            np.array(dones), dtype=torch.float32, device=self.device
        ).unsqueeze(-1)

        return states, actions, rewards, next_states, dones

    def ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer.

    Samples transitions with probability proportional to TD error.
    (Optional enhancement for future use)
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames to anneal beta to 1.0
            device: Torch device for tensors
        """
        super().__init__(capacity, device)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0

        # Priority storage
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def push(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        done: bool,
    ):
        """Store transition with max priority."""
        super().push(states, actions, rewards, next_states, done)
        self.priorities.append(self.max_priority)

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch with prioritized sampling.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        self.frame += 1

        # Compute sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        # Get transitions
        transitions = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Compute importance sampling weights
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(-1)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD errors.

        Args:
            indices: Indices of sampled transitions
            td_errors: TD errors for each transition
        """
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


if __name__ == "__main__":
    # Test replay buffer
    n_agents = 4
    obs_dim = 10
    action_dim = 2

    buffer = ReplayBuffer(capacity=1000)

    # Add some transitions
    for _ in range(100):
        states = np.random.randn(n_agents, obs_dim)
        actions = np.random.randn(n_agents, action_dim)
        rewards = np.random.randn(n_agents)
        next_states = np.random.randn(n_agents, obs_dim)
        done = random.random() < 0.1

        buffer.push(states, actions, rewards, next_states, done)

    print(f"Buffer size: {len(buffer)}")
    print(f"Ready for batch of 32: {buffer.ready(32)}")

    # Sample batch
    batch = buffer.sample(32)
    states, actions, rewards, next_states, dones = batch

    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Next states shape: {next_states.shape}")
    print(f"Dones shape: {dones.shape}")

    print("Replay buffer tests passed!")
