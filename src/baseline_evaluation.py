"""
Baseline Evaluation Script for MADDPG Traffic Light Optimization.

Compares trained RL model against fixed-timing baseline to verify
that learning is actually improving traffic control.

Usage:
    uv run python src/baseline_evaluation.py --model results/2x2/final_model.pt
    uv run python src/baseline_evaluation.py --model results/2x2/checkpoint_ep100.pt --episodes 5
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import traci
from sumolib import checkBinary

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env import TrafficEnv
from src.maddpg import MADDPG
from src.utils import load_config, ensure_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare RL vs Fixed-Timing Baseline")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="2x2",
        choices=["2x2", "3x3"],
        help="Scenario to evaluate (default: 2x2)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Episodes per flow file (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for comparison report",
    )

    return parser.parse_args()


def run_fixed_timing(
    scenario: str,
    flow_file: str,
    episode_length: int = 3600,
    decision_interval: int = 10,
) -> Dict[str, float]:
    """
    Run SUMO with default fixed-timing traffic light control.

    The network's built-in traffic light programs are used without
    any RL intervention.

    Args:
        scenario: Grid size ("2x2" or "3x3")
        flow_file: Flow file name
        episode_length: Simulation length in seconds
        decision_interval: Observation interval

    Returns:
        Dictionary of metrics (throughput, waiting_time)
    """
    scenario_path = Path("scenarios") / scenario / f"{scenario}_vietnamese"
    network_file = scenario_path / "network.net.xml"
    route_file = scenario_path / flow_file

    sumo_binary = checkBinary("sumo")
    sumo_cmd = [
        sumo_binary,
        "-n", str(network_file),
        "-r", str(route_file),
        "--step-length", "1.0",
        "--no-step-log", "true",
        "-W", "true",
        "--time-to-teleport", "300",
    ]

    try:
        traci.close()
    except traci.exceptions.FatalTraCIError:
        pass

    traci.start(sumo_cmd)

    time = 0
    cumulative_waiting_time = 0
    total_throughput = 0
    traffic_light_ids = list(traci.trafficlight.getIDList())

    # Run one initialization step
    traci.simulationStep()
    time += 1.0

    # Run simulation with fixed timing (no intervention)
    while time < episode_length:
        # Just advance simulation - let default programs run
        for _ in range(decision_interval):
            traci.simulationStep()
            time += 1.0

        # Collect metrics
        for tl_id in traffic_light_ids:
            for lane in traci.trafficlight.getControlledLanes(tl_id):
                cumulative_waiting_time += traci.lane.getLastStepHaltingNumber(lane)

        total_throughput += traci.simulation.getArrivedNumber()

    traci.close()

    return {
        "throughput": total_throughput,
        "waiting_time": cumulative_waiting_time,
    }


def run_rl_model(
    maddpg: MADDPG,
    scenario: str,
    flow_file: str,
    episode_length: int = 3600,
    decision_interval: int = 10,
) -> Dict[str, float]:
    """
    Run trained RL model on a specific flow file.

    Args:
        maddpg: Trained MADDPG model
        scenario: Grid size
        flow_file: Flow file name
        episode_length: Simulation length
        decision_interval: Decision interval

    Returns:
        Dictionary of metrics
    """
    env = TrafficEnv(
        scenario=scenario,
        mode="binary",
        decision_interval=decision_interval,
        episode_length=episode_length,
        training_flows=[flow_file],  # Force specific flow file
    )

    state = env.reset()
    done = False
    cumulative_waiting_time = 0
    total_reward = 0.0

    # Set model to evaluation mode (no exploration)
    original_eps = maddpg.eps
    maddpg.eps = 0.0  # Greedy actions only

    while not done:
        actions, _ = maddpg.select_actions(state)
        next_state, rewards, done = env.step(actions)

        cumulative_waiting_time += env._get_total_waiting()
        total_reward += rewards.mean()
        state = next_state

    metrics = env.get_metrics()
    env.close()

    # Restore epsilon
    maddpg.eps = original_eps

    return {
        "throughput": metrics["total_throughput"],
        "waiting_time": cumulative_waiting_time,
        "reward": total_reward,
    }


def run_comparison(
    maddpg: MADDPG,
    scenario: str,
    flow_files: List[str],
    episodes_per_flow: int = 3,
) -> Tuple[Dict, Dict]:
    """
    Run full comparison between RL and fixed-timing baseline.

    Args:
        maddpg: Trained MADDPG model
        scenario: Grid size
        flow_files: List of flow files to test
        episodes_per_flow: Number of episodes per flow file

    Returns:
        Tuple of (rl_results, baseline_results) dictionaries
    """
    rl_results = {flow: [] for flow in flow_files}
    baseline_results = {flow: [] for flow in flow_files}

    print("\n" + "=" * 60)
    print("Running Comparison: RL vs Fixed-Timing Baseline")
    print("=" * 60)

    for flow_file in flow_files:
        flow_name = Path(flow_file).stem
        print(f"\nFlow: {flow_name}")
        print("-" * 40)

        for ep in range(episodes_per_flow):
            # Run fixed-timing baseline
            baseline_metrics = run_fixed_timing(scenario, flow_file)
            baseline_results[flow_file].append(baseline_metrics)

            # Run RL model
            rl_metrics = run_rl_model(maddpg, scenario, flow_file)
            rl_results[flow_file].append(rl_metrics)

            # Print episode comparison
            tp_diff = rl_metrics["throughput"] - baseline_metrics["throughput"]
            wt_diff = baseline_metrics["waiting_time"] - rl_metrics["waiting_time"]

            print(f"  Ep {ep + 1}: "
                  f"RL={rl_metrics['throughput']:4d} tp, {rl_metrics['waiting_time']:7.0f} wt | "
                  f"Fixed={baseline_metrics['throughput']:4d} tp, {baseline_metrics['waiting_time']:7.0f} wt | "
                  f"Diff: {tp_diff:+4d} tp, {wt_diff:+7.0f} wt")

    return rl_results, baseline_results


def generate_report(
    rl_results: Dict,
    baseline_results: Dict,
    output_path: str = None,
) -> str:
    """
    Generate comparison report.

    Args:
        rl_results: RL model results by flow file
        baseline_results: Fixed-timing results by flow file
        output_path: Optional path to save report

    Returns:
        Report string
    """
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("COMPARISON REPORT: RL vs Fixed-Timing Baseline")
    lines.append("=" * 70)

    # Aggregate metrics
    all_rl_throughputs = []
    all_rl_waiting = []
    all_baseline_throughputs = []
    all_baseline_waiting = []

    lines.append("\n--- Per-Flow Summary ---\n")
    lines.append(f"{'Flow':<25} {'RL Throughput':>15} {'Baseline':>12} {'Diff':>10}")
    lines.append(f"{'':<25} {'RL Waiting':>15} {'Baseline':>12} {'Diff':>10}")
    lines.append("-" * 70)

    for flow_file in rl_results.keys():
        flow_name = Path(flow_file).stem.replace("flows_training_", "")

        rl_tp = np.mean([r["throughput"] for r in rl_results[flow_file]])
        rl_wt = np.mean([r["waiting_time"] for r in rl_results[flow_file]])
        base_tp = np.mean([r["throughput"] for r in baseline_results[flow_file]])
        base_wt = np.mean([r["waiting_time"] for r in baseline_results[flow_file]])

        all_rl_throughputs.extend([r["throughput"] for r in rl_results[flow_file]])
        all_rl_waiting.extend([r["waiting_time"] for r in rl_results[flow_file]])
        all_baseline_throughputs.extend([r["throughput"] for r in baseline_results[flow_file]])
        all_baseline_waiting.extend([r["waiting_time"] for r in baseline_results[flow_file]])

        tp_diff = rl_tp - base_tp
        wt_diff = base_wt - rl_wt  # Positive = RL is better (less waiting)

        lines.append(f"{flow_name:<25} {rl_tp:>15.1f} {base_tp:>12.1f} {tp_diff:>+10.1f}")
        lines.append(f"{'':<25} {rl_wt:>15.1f} {base_wt:>12.1f} {wt_diff:>+10.1f}")
        lines.append("")

    # Overall summary
    lines.append("\n--- Overall Summary ---\n")

    mean_rl_tp = np.mean(all_rl_throughputs)
    mean_rl_wt = np.mean(all_rl_waiting)
    mean_base_tp = np.mean(all_baseline_throughputs)
    mean_base_wt = np.mean(all_baseline_waiting)

    tp_improvement = ((mean_rl_tp - mean_base_tp) / mean_base_tp) * 100
    wt_improvement = ((mean_base_wt - mean_rl_wt) / mean_base_wt) * 100

    lines.append(f"Mean Throughput:  RL={mean_rl_tp:.1f}  Baseline={mean_base_tp:.1f}  ({tp_improvement:+.1f}%)")
    lines.append(f"Mean Waiting:     RL={mean_rl_wt:.1f}  Baseline={mean_base_wt:.1f}  ({wt_improvement:+.1f}%)")

    lines.append("\n--- Verdict ---\n")
    if tp_improvement > 5 and wt_improvement > 5:
        lines.append("RL OUTPERFORMS fixed-timing baseline (both throughput and waiting time)")
    elif tp_improvement > 0 or wt_improvement > 0:
        lines.append("RL shows MIXED results compared to baseline")
        if tp_improvement > 0:
            lines.append(f"  + Throughput improved by {tp_improvement:.1f}%")
        else:
            lines.append(f"  - Throughput decreased by {-tp_improvement:.1f}%")
        if wt_improvement > 0:
            lines.append(f"  + Waiting time reduced by {wt_improvement:.1f}%")
        else:
            lines.append(f"  - Waiting time increased by {-wt_improvement:.1f}%")
    else:
        lines.append("WARNING: RL UNDERPERFORMS fixed-timing baseline")
        lines.append("Consider: more training, hyperparameter tuning, or reward shaping")

    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)

    if output_path:
        ensure_dir(os.path.dirname(output_path))
        with open(output_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")

    return report


def main():
    """Main evaluation function."""
    if "SUMO_HOME" not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")

    args = parse_args()

    # Load config
    config = load_config("configs/default.yaml")

    # Update scenario
    if args.scenario == "2x2":
        config.n_agents = 4
    elif args.scenario == "3x3":
        config.n_agents = 9

    # Initialize environment to get obs_dim
    env = TrafficEnv(scenario=args.scenario, mode="binary")
    state = env.reset()
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    n_agents = env.n_agents
    env.close()

    print(f"\nLoading model from: {args.model}")
    print(f"Scenario: {args.scenario}")
    print(f"Agents: {n_agents}, Obs dim: {obs_dim}, Action dim: {action_dim}")

    # Initialize and load MADDPG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maddpg = MADDPG(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )
    maddpg.load(args.model)
    print(f"Model loaded. Epsilon: {maddpg.eps:.4f}")

    # Flow files to test
    flow_files = config.training_flows

    # Run comparison
    rl_results, baseline_results = run_comparison(
        maddpg=maddpg,
        scenario=args.scenario,
        flow_files=flow_files,
        episodes_per_flow=args.episodes,
    )

    # Generate report
    output_path = args.output or f"results/{args.scenario}/baseline_comparison.txt"
    report = generate_report(rl_results, baseline_results, output_path)
    print(report)


if __name__ == "__main__":
    main()
