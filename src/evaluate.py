"""
Quick evaluation script to check model behavior.

Checks:
- Action distribution (is it biased to one action?)
- Phase switching frequency
- Per-agent behavior
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env import TrafficEnv
from src.maddpg import MADDPG
from src.utils import load_config


def evaluate_model(checkpoint_path: str = None, episodes: int = 3, verbose: bool = True):
    """Evaluate model behavior."""

    if "SUMO_HOME" not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")

    config = load_config("configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize environment
    env = TrafficEnv(
        scenario=config.scenario,
        mode="binary",
        decision_interval=config.decision_interval,
        episode_length=config.episode_length,
        step_length=config.step_length,
        training_flows=["flows_training_balanced.rou.xml"],  # Use balanced for consistency
        min_phase_duration=config.min_phase_duration,
        switch_penalty=config.switch_penalty,
    )

    # Get dimensions
    state = env.reset()
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    env.close()

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
        device=device,
    )

    # Load checkpoint if provided
    if checkpoint_path and checkpoint_path != "none" and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        maddpg.load(checkpoint_path)
        maddpg.eps = 0.0  # No exploration during evaluation
    else:
        print("No checkpoint - using random/untrained policy (high epsilon)")
        maddpg.eps = 0.9  # Random for comparison

    print(f"\nEvaluating with epsilon={maddpg.eps}")
    print(f"Number of agents: {env.n_agents}")
    print(f"Traffic lights: {env.traffic_light_ids if hasattr(env, 'traffic_light_ids') and env.traffic_light_ids else 'TBD'}")
    print("=" * 60)

    # Track statistics
    all_actions = defaultdict(list)  # agent_idx -> list of actions
    all_action_probs = defaultdict(list)  # agent_idx -> list of action probs
    phase_switches = defaultdict(int)  # agent_idx -> count
    episode_rewards = []
    episode_throughputs = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        step = 0
        ep_reward = 0
        prev_actions = None

        print(f"\n[Episode {ep+1}] Flow: {env.current_flow_file}")
        print(f"Traffic lights: {env.traffic_light_ids}")

        while not done:
            step += 1

            # Get actions
            actions, action_probs = maddpg.select_actions(state)

            # Track actions
            for i, (action, probs) in enumerate(zip(actions, action_probs)):
                all_actions[i].append(action)
                all_action_probs[i].append(probs)

                # Track phase switches
                if prev_actions is not None and action != prev_actions[i]:
                    phase_switches[i] += 1

            # Step environment
            next_state, rewards, done = env.step(actions)
            ep_reward += rewards.mean()

            # Print first few steps in detail
            if verbose and step <= 5:
                print(f"  Step {step}:")
                for i in range(env.n_agents):
                    print(f"    Agent {i} ({env.traffic_light_ids[i]}): action={actions[i]}, probs={action_probs[i]}")
                print(f"    Rewards: {rewards}")

            prev_actions = actions
            state = next_state

        metrics = env.get_metrics()
        episode_rewards.append(ep_reward)
        episode_throughputs.append(metrics["total_throughput"])

        print(f"  -> Reward: {ep_reward:.2f}, Throughput: {metrics['total_throughput']}, Steps: {step}")

    env.close()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("BEHAVIOR ANALYSIS")
    print("=" * 60)

    print("\n1. ACTION DISTRIBUTION (per agent):")
    for i in range(env.n_agents):
        actions = all_actions[i]
        action_0_pct = actions.count(0) / len(actions) * 100
        action_1_pct = actions.count(1) / len(actions) * 100

        bias = "BIASED to 0" if action_0_pct > 80 else "BIASED to 1" if action_1_pct > 80 else "BALANCED"
        print(f"  Agent {i}: Action 0: {action_0_pct:.1f}%, Action 1: {action_1_pct:.1f}% [{bias}]")

    print("\n2. AVERAGE ACTION PROBABILITIES (from network):")
    for i in range(env.n_agents):
        probs = np.array(all_action_probs[i])
        avg_probs = probs.mean(axis=0)
        print(f"  Agent {i}: P(action=0)={avg_probs[0]:.3f}, P(action=1)={avg_probs[1]:.3f}")

    print("\n3. PHASE SWITCHING FREQUENCY:")
    total_steps = len(all_actions[0])
    for i in range(env.n_agents):
        switch_rate = phase_switches[i] / total_steps * 100
        behavior = "TOO FREQUENT" if switch_rate > 50 else "TOO STATIC" if switch_rate < 5 else "REASONABLE"
        print(f"  Agent {i}: {phase_switches[i]} switches ({switch_rate:.1f}% of steps) [{behavior}]")

    print("\n4. OVERALL PERFORMANCE:")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"  Mean throughput: {np.mean(episode_throughputs):.1f} vehicles")

    return {
        "action_distributions": {i: all_actions[i] for i in range(env.n_agents)},
        "phase_switches": dict(phase_switches),
        "rewards": episode_rewards,
        "throughputs": episode_throughputs,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    args = parser.parse_args()

    # If no checkpoint specified, try to find the latest one
    if args.checkpoint is None:
        results_dir = Path("results/2x2")
        if results_dir.exists():
            checkpoints = list(results_dir.glob("checkpoint_ep*.pt"))
            if checkpoints:
                # Get latest by episode number
                latest = max(checkpoints, key=lambda p: int(p.stem.split("_ep")[-1]))
                args.checkpoint = str(latest)
                print(f"Using latest checkpoint: {args.checkpoint}")

    evaluate_model(args.checkpoint, args.episodes)
