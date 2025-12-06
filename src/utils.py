"""
Utility functions for MADDPG Traffic Light Optimization.

Includes:
- Configuration loading
- Metrics computation and CSV export
- Logging utilities
- Visualization (training curves, waiting time, throughput)
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import yaml
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Config:
    """Configuration dataclass for MADDPG training."""

    # Environment
    scenario: str = "2x2"
    decision_interval: int = 10
    episode_length: int = 3600
    step_length: float = 1.0
    min_phase_duration: int = 10
    switch_penalty: float = 0.05  # Now scaled for density-normalized rewards

    # MADDPG
    lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 100000  # Larger buffer for better diversity
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])

    # Prioritized Experience Replay
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 150000

    # Exploration (linear episode-based decay)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 400

    # Training
    n_episodes: int = 500
    checkpoint_interval: int = 10
    log_interval: int = 1
    render: bool = False

    # Scenario specifics
    n_agents: int = 4
    scenario_path: str = "scenarios/2x2/2x2_vietnamese"

    # Training flows
    training_flows: List[str] = field(default_factory=lambda: [
        "flows_training_balanced.rou.xml",
        "flows_training_light.rou.xml",
        "flows_training_heavy.rou.xml",
        "flows_training_ns_dominant.rou.xml",
        "flows_training_ew_dominant.rou.xml",
        "flows_training_time_varying.rou.xml",
    ])


def load_config(config_path: str = "configs/default.yaml") -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config dataclass with loaded values
    """
    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Flatten nested config
    flat_config = {}

    if "environment" in yaml_config:
        flat_config.update(yaml_config["environment"])

    if "maddpg" in yaml_config:
        flat_config.update(yaml_config["maddpg"])

    if "per" in yaml_config:
        flat_config.update(yaml_config["per"])

    if "exploration" in yaml_config:
        flat_config.update(yaml_config["exploration"])

    if "training" in yaml_config:
        flat_config.update(yaml_config["training"])

    if "training_flows" in yaml_config:
        flat_config["training_flows"] = yaml_config["training_flows"]

    # Get scenario-specific settings
    scenario = flat_config.get("scenario", "2x2")
    if "scenarios" in yaml_config and scenario in yaml_config["scenarios"]:
        scenario_config = yaml_config["scenarios"][scenario]
        flat_config["n_agents"] = scenario_config.get("n_agents", 4)
        flat_config["scenario_path"] = scenario_config.get("scenario_path", f"scenarios/{scenario}/{scenario}_vietnamese")

    return Config(**{k: v for k, v in flat_config.items() if hasattr(Config, k)})


class MetricsTracker:
    """Track and compute training metrics."""

    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []
        self.epsilons: List[float] = []
        self.throughputs: List[int] = []
        self.waiting_times: List[float] = []
        self.flow_files: List[str] = []

        # Per-episode accumulators
        self._current_rewards: List[float] = []
        self._current_actor_losses: List[float] = []
        self._current_critic_losses: List[float] = []

    def step(
        self,
        reward: float,
        actor_loss: Optional[float] = None,
        critic_loss: Optional[float] = None,
    ):
        """Record step metrics."""
        self._current_rewards.append(reward)
        if actor_loss is not None:
            self._current_actor_losses.append(actor_loss)
        if critic_loss is not None:
            self._current_critic_losses.append(critic_loss)

    def end_episode(
        self,
        episode_length: int,
        epsilon: float,
        throughput: int = 0,
        waiting_time: float = 0.0,
        flow_file: str = "",
    ):
        """End episode and compute metrics."""
        # Episode reward
        episode_reward = sum(self._current_rewards)
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.epsilons.append(epsilon)
        self.throughputs.append(throughput)
        self.waiting_times.append(waiting_time)
        self.flow_files.append(flow_file)

        # Losses
        if self._current_actor_losses:
            self.actor_losses.append(np.mean(self._current_actor_losses))
        if self._current_critic_losses:
            self.critic_losses.append(np.mean(self._current_critic_losses))

        # Reset accumulators
        self._current_rewards = []
        self._current_actor_losses = []
        self._current_critic_losses = []

    def get_summary(self, last_n: int = 10) -> Dict[str, float]:
        """Get summary of recent metrics."""
        summary = {}

        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-last_n:]
            summary["mean_reward"] = np.mean(recent_rewards)
            summary["std_reward"] = np.std(recent_rewards)

        if self.episode_lengths:
            summary["mean_length"] = np.mean(self.episode_lengths[-last_n:])

        if self.actor_losses:
            summary["mean_actor_loss"] = np.mean(self.actor_losses[-last_n:])

        if self.critic_losses:
            summary["mean_critic_loss"] = np.mean(self.critic_losses[-last_n:])

        if self.throughputs:
            summary["mean_throughput"] = np.mean(self.throughputs[-last_n:])

        if self.waiting_times:
            summary["mean_waiting_time"] = np.mean(self.waiting_times[-last_n:])

        return summary


def save_metrics_to_csv(metrics: MetricsTracker, save_path: str):
    """
    Save all episode metrics to CSV file.

    Args:
        metrics: MetricsTracker with recorded data
        save_path: Path to save CSV file
    """
    ensure_dir(os.path.dirname(save_path))

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'reward', 'throughput', 'waiting_time',
            'episode_length', 'epsilon', 'actor_loss', 'critic_loss', 'flow_file'
        ])

        for i in range(len(metrics.episode_rewards)):
            writer.writerow([
                i + 1,
                metrics.episode_rewards[i] if i < len(metrics.episode_rewards) else 0,
                metrics.throughputs[i] if i < len(metrics.throughputs) else 0,
                metrics.waiting_times[i] if i < len(metrics.waiting_times) else 0,
                metrics.episode_lengths[i] if i < len(metrics.episode_lengths) else 0,
                metrics.epsilons[i] if i < len(metrics.epsilons) else 0,
                metrics.actor_losses[i] if i < len(metrics.actor_losses) else 0,
                metrics.critic_losses[i] if i < len(metrics.critic_losses) else 0,
                metrics.flow_files[i] if i < len(metrics.flow_files) else '',
            ])


def plot_training_curves(
    metrics: MetricsTracker,
    save_path: str,
    title: str = "MADDPG Training",
):
    """
    Plot training curves (4-panel: rewards, throughput, losses, epsilon).

    Args:
        metrics: MetricsTracker with recorded data
        save_path: Path to save the figure
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)

    # Episode rewards
    ax = axes[0, 0]
    ax.plot(metrics.episode_rewards, alpha=0.7, label="Episode Reward")
    if len(metrics.episode_rewards) > 10:
        smoothed = np.convolve(
            metrics.episode_rewards,
            np.ones(10) / 10,
            mode="valid"
        )
        ax.plot(range(9, len(metrics.episode_rewards)), smoothed, label="Smoothed (10)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Throughput
    ax = axes[0, 1]
    ax.plot(metrics.throughputs, alpha=0.7)
    if len(metrics.throughputs) > 10:
        smoothed = np.convolve(metrics.throughputs, np.ones(10) / 10, mode="valid")
        ax.plot(range(9, len(metrics.throughputs)), smoothed, linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Vehicles")
    ax.set_title("Throughput (Arrived Vehicles)")
    ax.grid(True, alpha=0.3)

    # Losses
    ax = axes[1, 0]
    if metrics.actor_losses:
        ax.plot(metrics.actor_losses, label="Actor Loss", alpha=0.7)
    if metrics.critic_losses:
        ax.plot(metrics.critic_losses, label="Critic Loss", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.set_title("Training Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Epsilon
    ax = axes[1, 1]
    ax.plot(metrics.epsilons, alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate (ε-greedy)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_key_metrics(
    metrics: MetricsTracker,
    save_path: str,
    title: str = "Key Metrics",
):
    """
    Plot dedicated waiting time and throughput metrics.

    Args:
        metrics: MetricsTracker with recorded data
        save_path: Path to save the figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Throughput
    ax = axes[0]
    ax.plot(metrics.throughputs, alpha=0.5, color='green', label='Raw')
    if len(metrics.throughputs) > 10:
        smoothed = np.convolve(metrics.throughputs, np.ones(10) / 10, mode='valid')
        ax.plot(range(9, len(metrics.throughputs)), smoothed, color='darkgreen', linewidth=2, label='Smoothed (10)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Vehicles Arrived')
    ax.set_title('Throughput per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Waiting Time
    ax = axes[1]
    ax.plot(metrics.waiting_times, alpha=0.5, color='red', label='Raw')
    if len(metrics.waiting_times) > 10:
        smoothed = np.convolve(metrics.waiting_times, np.ones(10) / 10, mode='valid')
        ax.plot(range(9, len(metrics.waiting_times)), smoothed, color='darkred', linewidth=2, label='Smoothed (10)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Waiting Time')
    ax.set_title('Waiting Time per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def ensure_dir(path: str):
    """Ensure directory exists."""
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


class Logger:
    """Simple logger for training."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        if log_file:
            ensure_dir(os.path.dirname(log_file))

    def log(self, message: str, also_print: bool = True):
        """Log a message."""
        if also_print:
            print(message)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")

    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        epsilon: float,
        throughput: int,
        waiting_time: float,
        flow_file: str,
        duration: float,
    ):
        """Log episode summary."""
        msg = (
            f"[Ep {episode:4d}] "
            f"Reward: {reward:8.2f} | "
            f"Steps: {length:4d} | "
            f"Throughput: {throughput:4d} | "
            f"Waiting: {waiting_time:8.0f} | "
            f"Eps: {epsilon:.3f} | "
            f"Flow: {flow_file} | "
            f"Time: {format_time(duration)}"
        )
        self.log(msg)


if __name__ == "__main__":
    # Test config loading
    config = load_config("configs/default.yaml")
    print(f"Scenario: {config.scenario}")
    print(f"N agents: {config.n_agents}")
    print(f"Episodes: {config.n_episodes}")
    print(f"Buffer size: {config.buffer_size}")
    print(f"Epsilon start: {config.eps_start}")

    # Test metrics tracker
    tracker = MetricsTracker()
    for ep in range(20):
        for step in range(10):
            tracker.step(np.random.randn(), 0.1, 0.5)
        tracker.end_episode(
            episode_length=100,
            epsilon=0.9 - ep * 0.04,
            throughput=50 + ep * 5,
            waiting_time=1000 - ep * 40,
            flow_file="test.xml"
        )

    summary = tracker.get_summary()
    print(f"Summary: {summary}")

    # Test CSV export
    save_metrics_to_csv(tracker, "/tmp/test_metrics.csv")
    print("CSV saved to /tmp/test_metrics.csv")

    # Test plotting
    plot_training_curves(tracker, "/tmp/test_curves.png", "Test Training")
    print("Curves saved to /tmp/test_curves.png")

    plot_key_metrics(tracker, "/tmp/test_key_metrics.png", "Test Key Metrics")
    print("Key metrics saved to /tmp/test_key_metrics.png")

    print("Utils tests passed!")
