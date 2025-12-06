"""
Training script for MADDPG Traffic Light Optimization.

Follows sumo-marl reference implementation with metrics logging.

Usage:
    uv run python src/train.py --scenario 2x2 --episodes 100
    uv run python src/train.py --scenario 3x3 --render
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env import TrafficEnv
from src.maddpg import MADDPG
from src.utils import (
    load_config,
    MetricsTracker,
    Logger,
    plot_training_curves,
    plot_key_metrics,
    save_metrics_to_csv,
    ensure_dir,
    format_time,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MADDPG for Traffic Light Control")

    parser.add_argument(
        "--scenario",
        type=str,
        default="2x2",
        choices=["2x2", "3x3"],
        help="Scenario to train on (default: 2x2)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes (overrides config)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render SUMO GUI during training",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device to use (default: auto)",
    )

    return parser.parse_args()


def main():
    """Main training loop."""
    # Check SUMO_HOME
    if "SUMO_HOME" not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")

    # Parse arguments
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with args
    if args.scenario:
        config.scenario = args.scenario
        if args.scenario == "2x2":
            config.n_agents = 4
            config.scenario_path = "scenarios/2x2/2x2_vietnamese"
        elif args.scenario == "3x3":
            config.n_agents = 9
            config.scenario_path = "scenarios/3x3/3x3_vietnamese"

    if args.episodes:
        config.n_episodes = args.episodes

    if args.render:
        config.render = True

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup directories
    results_dir = f"results/{config.scenario}"
    ensure_dir(results_dir)

    # Initialize logger
    logger = Logger(log_file=f"{results_dir}/training.log")
    logger.log(f"=" * 60)
    logger.log(f"MADDPG Traffic Light Optimization (Improved Stability)")
    logger.log(f"=" * 60)
    logger.log(f"Scenario: {config.scenario}")
    logger.log(f"Episodes: {config.n_episodes}")
    logger.log(f"Device: {device}")
    logger.log(f"Render: {config.render}")
    logger.log(f"Buffer size: {config.buffer_size}")
    logger.log(f"PER: {config.use_per} (alpha={config.per_alpha})")
    logger.log(f"Epsilon: {config.eps_start} -> {config.eps_end} (linear over {config.eps_decay_episodes} eps)")
    logger.log(f"Reward: density-normalized (bounded [-1, 0])")
    logger.log(f"=" * 60)

    # Initialize environment
    env = TrafficEnv(
        scenario=config.scenario,
        mode="gui" if config.render else "binary",
        decision_interval=config.decision_interval,
        episode_length=config.episode_length,
        step_length=config.step_length,
        training_flows=config.training_flows,
        min_phase_duration=config.min_phase_duration,
        # switch_penalty now hardcoded in env.py as 0.05 for density-normalized rewards
    )

    # Get obs_dim from environment after reset (dynamically determined)
    state = env.reset()
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    env.close()

    logger.log(f"Observation dimension: {obs_dim}")
    logger.log(f"Action dimension: {action_dim}")
    logger.log(f"Number of agents: {env.n_agents}")

    # Initialize MADDPG
    maddpg = MADDPG(
        n_agents=env.n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=config.lr,
        gamma=config.gamma,
        tau=config.tau,
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        hidden=config.hidden_sizes,
        eps_start=config.eps_start,
        eps_end=config.eps_end,
        eps_decay_episodes=config.eps_decay_episodes,
        use_per=config.use_per,
        per_alpha=config.per_alpha,
        per_beta_start=config.per_beta_start,
        per_beta_frames=config.per_beta_frames,
        device=device,
    )

    # Load checkpoint if provided
    start_episode = 0
    if args.checkpoint:
        logger.log(f"Loading checkpoint from {args.checkpoint}")
        maddpg.load(args.checkpoint)
        # Try to extract episode number from filename
        try:
            start_episode = int(Path(args.checkpoint).stem.split("_ep")[-1])
        except (ValueError, IndexError):
            pass

    # Initialize metrics tracker
    metrics = MetricsTracker()

    # Training loop
    training_start = time.time()
    training_started = False

    logger.log(f"\nStarting training from episode {start_episode + 1}...")

    try:
        for episode in range(start_episode, config.n_episodes):
            episode_start = time.time()

            # Reset environment (randomly selects flow file)
            state = env.reset()
            done = False
            step_count = 0
            episode_reward = 0.0

            # Track cumulative metrics for this episode
            episode_waiting_time = 0
            episode_throughput = 0

            while not done:
                step_count += 1

                # Select actions for all agents
                actions, action_probs = maddpg.select_actions(state)

                # Execute actions
                next_state, rewards, done = env.step(actions)

                # Store transition
                maddpg.store_transition(state, action_probs, rewards, next_state, done)

                # Train if ready
                if maddpg.ready_to_train():
                    if not training_started:
                        logger.log("\n[Training started - replay buffer ready!]\n")
                        training_started = True

                    actor_loss, critic_loss = maddpg.update_all()
                    metrics.step(rewards.mean(), actor_loss, critic_loss)
                else:
                    metrics.step(rewards.mean())

                # Update cumulative metrics
                episode_waiting_time += env._get_total_waiting()
                episode_throughput = env.total_throughput  # Use env's cumulative count

                # Update state
                state = next_state
                episode_reward += rewards.mean()

            # End episode
            episode_duration = time.time() - episode_start
            env_metrics = env.get_metrics()

            # Update epsilon (linear episode-based decay)
            maddpg.update_eps(episode + 1)

            metrics.end_episode(
                episode_length=step_count,
                epsilon=maddpg.eps,
                throughput=env_metrics["total_throughput"],
                waiting_time=episode_waiting_time,
                flow_file=env_metrics["flow_file"],
            )

            # Log episode
            logger.log_episode(
                episode=episode + 1,
                reward=episode_reward,
                length=step_count,
                epsilon=maddpg.eps,
                throughput=env_metrics["total_throughput"],
                waiting_time=episode_waiting_time,
                flow_file=Path(env_metrics["flow_file"]).stem,
                duration=episode_duration,
            )

            # Checkpoint
            if (episode + 1) % config.checkpoint_interval == 0:
                checkpoint_path = f"{results_dir}/checkpoint_ep{episode + 1}.pt"
                maddpg.save(checkpoint_path)
                logger.log(f"[Checkpoint saved: {checkpoint_path}]")

                # Save metrics to CSV
                csv_path = f"{results_dir}/metrics.csv"
                save_metrics_to_csv(metrics, csv_path)

                # Plot training curves
                plot_training_curves(
                    metrics,
                    f"{results_dir}/training_curves.png",
                    title=f"MADDPG Training - {config.scenario}",
                )

                # Plot key metrics (waiting time + throughput)
                plot_key_metrics(
                    metrics,
                    f"{results_dir}/waiting_throughput.png",
                    title=f"Key Metrics - {config.scenario}",
                )

    except KeyboardInterrupt:
        logger.log("\n[Training interrupted by user]")

    finally:
        # Close environment
        env.close()

        # Save final model
        final_path = f"{results_dir}/final_model.pt"
        maddpg.save(final_path)
        logger.log(f"\n[Final model saved: {final_path}]")

        # Save final metrics CSV
        csv_path = f"{results_dir}/metrics.csv"
        save_metrics_to_csv(metrics, csv_path)
        logger.log(f"[Metrics saved: {csv_path}]")

        # Final plots
        plot_training_curves(
            metrics,
            f"{results_dir}/training_curves_final.png",
            title=f"MADDPG Training - {config.scenario} (Final)",
        )
        plot_key_metrics(
            metrics,
            f"{results_dir}/waiting_throughput_final.png",
            title=f"Key Metrics - {config.scenario} (Final)",
        )

        # Training summary
        total_time = time.time() - training_start
        logger.log(f"\n{'=' * 60}")
        logger.log(f"Training Complete!")
        logger.log(f"Total time: {format_time(total_time)}")
        logger.log(f"Episodes: {len(metrics.episode_rewards)}")

        if metrics.episode_rewards:
            logger.log(f"Best reward: {max(metrics.episode_rewards):.2f}")
            logger.log(f"Final avg reward (10): {np.mean(metrics.episode_rewards[-10:]):.2f}")

        if metrics.throughputs:
            logger.log(f"Best throughput: {max(metrics.throughputs)}")
            logger.log(f"Final avg throughput (10): {np.mean(metrics.throughputs[-10:]):.2f}")

        if metrics.waiting_times:
            logger.log(f"Best waiting time: {min(metrics.waiting_times):.0f}")
            logger.log(f"Final avg waiting time (10): {np.mean(metrics.waiting_times[-10:]):.0f}")

        logger.log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
