"""
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Algorithm.

Follows sumo-marl reference implementation:
- ε-greedy exploration (not Gumbel-Softmax)
- Smaller replay buffer (5000)
- Simpler network architecture
"""

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .networks import Actor, Critic, soft_update, hard_update
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class MADDPG(nn.Module):
    """
    MADDPG algorithm for multi-agent traffic light control.

    Features:
    - Centralized training: Critics see all agents' observations and actions
    - Decentralized execution: Actors only use local observations
    - ε-greedy exploration (matches sumo-marl)
    - Soft target updates for stability
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int = 10,
        action_dim: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 100000,
        hidden: List[int] = None,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_episodes: int = 400,
        use_per: bool = True,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 150000,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize MADDPG with improved training stability.

        Args:
            n_agents: Number of agents
            obs_dim: Observation dimension per agent
            action_dim: Action dimension per agent
            lr: Learning rate for both actor and critic
            gamma: Discount factor
            tau: Soft update parameter
            batch_size: Training batch size
            buffer_size: Replay buffer capacity
            hidden: Hidden layer sizes (default: [64, 64])
            eps_start: Initial epsilon for ε-greedy
            eps_end: Minimum epsilon
            eps_decay_episodes: Episodes over which to decay epsilon linearly
            use_per: Whether to use Prioritized Experience Replay
            per_alpha: PER prioritization exponent
            per_beta_start: PER initial importance sampling weight
            per_beta_frames: PER frames to anneal beta to 1.0
            device: Torch device
        """
        super(MADDPG, self).__init__()

        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        if hidden is None:
            hidden = [64, 64]

        # Device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Linear episode-based epsilon decay
        self.eps = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = eps_decay_episodes
        self.current_episode = 0

        # Initialize networks
        self._init_networks(hidden)

        # Initialize optimizers (same lr for actor and critic)
        self._init_optimizers(lr)

        # Replay buffer (with optional PER)
        self.use_per = use_per
        if use_per:
            self.memory = PrioritizedReplayBuffer(
                capacity=buffer_size,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_frames=per_beta_frames,
                device=self.device,
            )
        else:
            self.memory = ReplayBuffer(capacity=buffer_size, device=self.device)

        # Loss function (no reduction for PER weighted loss)
        self.mse_loss = nn.MSELoss(reduction='none' if use_per else 'mean')

    def _init_networks(self, hidden: List[int]):
        """Initialize actor and critic networks for all agents."""
        # Actors (one per agent)
        self.actors = nn.ModuleList([
            Actor(self.obs_dim, self.action_dim, hidden)
            for _ in range(self.n_agents)
        ]).to(self.device)

        self.target_actors = nn.ModuleList([
            Actor(self.obs_dim, self.action_dim, hidden)
            for _ in range(self.n_agents)
        ]).to(self.device)

        # Critics (one per agent, centralized)
        state_dim = self.obs_dim * self.n_agents
        action_dim_total = self.action_dim * self.n_agents

        self.critics = nn.ModuleList([
            Critic(state_dim, action_dim_total, hidden)
            for _ in range(self.n_agents)
        ]).to(self.device)

        self.target_critics = nn.ModuleList([
            Critic(state_dim, action_dim_total, hidden)
            for _ in range(self.n_agents)
        ]).to(self.device)

        # Initialize target networks with same weights
        for i in range(self.n_agents):
            hard_update(self.actors[i], self.target_actors[i])
            hard_update(self.critics[i], self.target_critics[i])

    def _init_optimizers(self, lr: float):
        """Initialize optimizers for all networks."""
        self.actor_optimizers = [
            optim.Adam(self.actors[i].parameters(), lr=lr)
            for i in range(self.n_agents)
        ]
        self.critic_optimizers = [
            optim.Adam(self.critics[i].parameters(), lr=lr)
            for i in range(self.n_agents)
        ]

    def select_action(
        self,
        obs: np.ndarray,
        agent_idx: int,
    ) -> Tuple[int, np.ndarray]:
        """
        Select action for a single agent using ε-greedy (like sumo-marl).

        Args:
            obs: Agent's local observation
            agent_idx: Agent index

        Returns:
            Tuple of (action, action_probabilities)
        """
        obs_tensor = torch.from_numpy(obs).float().to(self.device)

        if random.random() < self.eps:
            # Random action (exploration)
            action = random.randint(0, self.action_dim - 1)
            action_prob = np.zeros(self.action_dim)
            action_prob[action] = 1.0
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                action_prob = self.actors[agent_idx](obs_tensor).cpu().numpy()
            action = int(np.argmax(action_prob))

        return action, action_prob

    def select_actions(
        self,
        states: np.ndarray,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Select actions for all agents.

        Args:
            states: All agents' observations, shape (n_agents, obs_dim)

        Returns:
            Tuple of (actions list, action_probs array)
        """
        actions = []
        action_probs = []

        for i in range(self.n_agents):
            action, probs = self.select_action(states[i], i)
            actions.append(action)
            action_probs.append(probs)

        return actions, np.array(action_probs)

    def store_transition(
        self,
        states: np.ndarray,
        action_probs: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        done: bool,
    ):
        """Store transition in replay buffer."""
        self.memory.push(states, action_probs, rewards, next_states, done)

    def ready_to_train(self) -> bool:
        """Check if enough samples for training."""
        return self.memory.ready(self.batch_size)

    def update(self, agent_idx: int) -> Tuple[float, float]:
        """
        Update networks for a single agent with optional PER support.

        Args:
            agent_idx: Index of agent to update

        Returns:
            Tuple of (actor_loss, critic_loss)
        """
        # Sample batch (with weights and indices if using PER)
        if self.use_per:
            states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(
                self.batch_size
            )
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(
                self.batch_size
            )
            weights = None
            indices = None

        # Reshape for processing
        batch_size = states.shape[0]
        states_per_agent = states.view(batch_size, self.n_agents, self.obs_dim)
        next_states_per_agent = next_states.view(batch_size, self.n_agents, self.obs_dim)

        # ============ Update Critic ============
        # Get current Q-value
        current_q = self.critics[agent_idx](states, actions)

        # Get target actions from all actors (using target actors)
        with torch.no_grad():
            next_actions = []
            for j in range(self.n_agents):
                # Each actor considers only its own observation
                next_action_probs = self.actors[j](next_states_per_agent[:, j])
                next_actions.append(next_action_probs)
            next_actions = torch.cat(next_actions, dim=1)

            # Get target Q-value
            next_q = self.target_critics[agent_idx](next_states, next_actions)

            # Compute TD target (agent's own reward)
            agent_rewards = rewards[:, agent_idx].unsqueeze(-1)
            target_q = agent_rewards + self.gamma * (1.0 - dones) * next_q

        # Critic loss (with optional PER weighting)
        td_errors = current_q - target_q
        if self.use_per:
            # Element-wise MSE weighted by importance sampling
            critic_loss_per_sample = td_errors.pow(2).squeeze()
            critic_loss = (weights * critic_loss_per_sample).mean()

            # Update priorities based on TD errors
            self.memory.update_priorities(indices, td_errors.detach().cpu().numpy().squeeze())
        else:
            critic_loss = td_errors.pow(2).mean()

        # Update critic
        self.critic_optimizers[agent_idx].zero_grad()
        critic_loss.backward()
        self.critic_optimizers[agent_idx].step()

        # ============ Update Actor ============
        # Get current actions from all actors
        current_actions = []
        for j in range(self.n_agents):
            action_probs = self.actors[j](states_per_agent[:, j])
            current_actions.append(action_probs)
        current_actions = torch.cat(current_actions, dim=1)

        # Actor loss (maximize Q-value = minimize -Q)
        actor_loss = -self.critics[agent_idx](states, current_actions).mean()

        # Update actor
        self.actor_optimizers[agent_idx].zero_grad()
        actor_loss.backward()
        self.actor_optimizers[agent_idx].step()

        # ============ Soft Update Targets ============
        soft_update(self.actors[agent_idx], self.target_actors[agent_idx], self.tau)
        soft_update(self.critics[agent_idx], self.target_critics[agent_idx], self.tau)

        return actor_loss.item(), critic_loss.item()

    def update_all(self) -> Tuple[float, float]:
        """
        Update all agents' networks.

        Returns:
            Tuple of (mean_actor_loss, mean_critic_loss)
        """
        actor_losses = []
        critic_losses = []

        for i in range(self.n_agents):
            actor_loss, critic_loss = self.update(i)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        return np.mean(actor_losses), np.mean(critic_losses)

    def update_eps(self, episode: int = None):
        """
        Update epsilon using linear episode-based decay.

        Args:
            episode: Current episode number (if provided, uses linear decay)
        """
        if episode is not None:
            self.current_episode = episode

            # Linear decay over eps_decay_episodes
            if self.current_episode < self.eps_decay_episodes:
                progress = self.current_episode / self.eps_decay_episodes
                self.eps = self.eps_start - (self.eps_start - self.eps_end) * progress
            else:
                self.eps = self.eps_end

    def save(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            "actors": [actor.state_dict() for actor in self.actors],
            "critics": [critic.state_dict() for critic in self.critics],
            "target_actors": [actor.state_dict() for actor in self.target_actors],
            "target_critics": [critic.state_dict() for critic in self.target_critics],
            "actor_optimizers": [opt.state_dict() for opt in self.actor_optimizers],
            "critic_optimizers": [opt.state_dict() for opt in self.critic_optimizers],
            "eps": self.eps,
            "current_episode": self.current_episode,
            "eps_start": self.eps_start,
            "eps_end": self.eps_end,
            "eps_decay_episodes": self.eps_decay_episodes,
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        for i in range(self.n_agents):
            self.actors[i].load_state_dict(checkpoint["actors"][i])
            self.critics[i].load_state_dict(checkpoint["critics"][i])
            self.target_actors[i].load_state_dict(checkpoint["target_actors"][i])
            self.target_critics[i].load_state_dict(checkpoint["target_critics"][i])
            self.actor_optimizers[i].load_state_dict(checkpoint["actor_optimizers"][i])
            self.critic_optimizers[i].load_state_dict(checkpoint["critic_optimizers"][i])

        self.eps = checkpoint.get("eps", self.eps_start)
        self.current_episode = checkpoint.get("current_episode", 0)


if __name__ == "__main__":
    # Test MADDPG
    n_agents = 4
    obs_dim = 10
    action_dim = 2

    maddpg = MADDPG(n_agents, obs_dim, action_dim)

    print(f"Device: {maddpg.device}")
    print(f"Number of agents: {maddpg.n_agents}")
    print(f"Initial epsilon: {maddpg.eps}")

    # Test action selection
    states = np.random.randn(n_agents, obs_dim).astype(np.float32)
    actions, action_probs = maddpg.select_actions(states)
    print(f"Actions: {actions}")
    print(f"Action probs shape: {action_probs.shape}")

    # Add some transitions
    for _ in range(100):
        states = np.random.randn(n_agents, obs_dim).astype(np.float32)
        actions, action_probs = maddpg.select_actions(states)
        rewards = np.random.randn(n_agents).astype(np.float32)
        next_states = np.random.randn(n_agents, obs_dim).astype(np.float32)
        done = random.random() < 0.1

        maddpg.store_transition(states, action_probs, rewards, next_states, done)
        maddpg.update_eps()

    print(f"Buffer size: {len(maddpg.memory)}")
    print(f"Ready to train: {maddpg.ready_to_train()}")
    print(f"Epsilon after 100 steps: {maddpg.eps:.4f}")

    # Test update
    if maddpg.ready_to_train():
        actor_loss, critic_loss = maddpg.update_all()
        print(f"Actor loss: {actor_loss:.4f}, Critic loss: {critic_loss:.4f}")

    print("MADDPG tests passed!")
