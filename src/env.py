"""
SUMO Traffic Light Environment for MADDPG

Follows sumo-marl reference implementation:
- 2-action space: phase 0 vs phase 1
- Simple state: lane counts + phase one-hot
- Simple reward: -halting_count
- Multi-flow training for generalization
"""

import os
import sys
import random
from pathlib import Path
from typing import Optional, List

import numpy as np
import traci
from sumolib import checkBinary


class TrafficEnv:
    """Multi-agent traffic light control environment using SUMO."""

    def __init__(
        self,
        scenario: str = "2x2",
        scenario_base_path: str = "scenarios",
        mode: str = "binary",
        decision_interval: int = 10,
        episode_length: int = 3600,
        step_length: float = 1.0,
        training_flows: Optional[List[str]] = None,
        min_phase_duration: int = 10,
        switch_penalty: float = 0.5,
    ):
        """
        Initialize the traffic environment.

        Args:
            scenario: Grid size ("2x2" or "3x3")
            scenario_base_path: Base path to scenario files
            mode: "gui" for visualization, "binary" for headless
            decision_interval: Seconds between agent decisions
            episode_length: Maximum episode length in seconds
            step_length: SUMO simulation step length
            training_flows: List of flow files for random selection
            min_phase_duration: Minimum seconds before phase can switch
            switch_penalty: Reward penalty for switching phases
        """
        self.scenario = scenario
        self.mode = mode
        self.decision_interval = decision_interval
        self.episode_length = episode_length
        self.step_length = step_length
        self.min_phase_duration = min_phase_duration
        self.switch_penalty = switch_penalty

        # Set up paths
        self.scenario_path = Path(scenario_base_path) / scenario / f"{scenario}_vietnamese"
        self.network_file = self.scenario_path / "network.net.xml"

        # Training flows for generalization (keep multi-flow)
        self.training_flows = training_flows or [
            "flows_training_balanced.rou.xml",
            "flows_training_light.rou.xml",
            "flows_training_heavy.rou.xml",
            "flows_training_ns_dominant.rou.xml",
            "flows_training_ew_dominant.rou.xml",
            "flows_training_time_varying.rou.xml",
        ]

        # SUMO binary
        if mode == "gui":
            self.sumo_binary = checkBinary("sumo-gui")
        else:
            self.sumo_binary = checkBinary("sumo")

        # Environment properties (set during reset)
        self.n_agents = 4 if scenario == "2x2" else 9
        self.traffic_light_ids: List[str] = []
        self.time = 0
        self.current_flow_file: str = ""

        # State and action dimensions (match sumo-marl)
        self.n_phase = 2  # Binary phase
        self.obs_dim = 10  # Will be dynamically set based on lanes + phase
        self.action_dim = 2  # Phase 0 or Phase 1

        # Metrics tracking
        self.cumulative_waiting_time = 0
        self.total_throughput = 0

        # Phase duration tracking (for min_phase_duration constraint)
        self.phase_timers: List[int] = []  # Time since last switch per agent
        self.switches_this_step: List[bool] = []  # Track switches for penalty

    def _get_sumo_cmd(self, flow_file: str) -> list:
        """Build SUMO command with the specified flow file."""
        route_file = self.scenario_path / flow_file
        return [
            self.sumo_binary,
            "-n", str(self.network_file),
            "-r", str(route_file),
            "--step-length", str(self.step_length),
            "--no-step-log", "true",
            "-W", "true",
            "--time-to-teleport", "300",
        ]

    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new episode.

        Returns:
            Initial state observation for all agents
        """
        # Close any existing connection
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass

        # Randomly select flow file for this episode
        self.current_flow_file = random.choice(self.training_flows)

        # Start SUMO
        sumo_cmd = self._get_sumo_cmd(self.current_flow_file)
        traci.start(sumo_cmd)

        # Initialize time and metrics
        self.time = 0
        self.cumulative_waiting_time = 0
        self.total_throughput = 0

        # Get traffic light IDs
        self.traffic_light_ids = list(traci.trafficlight.getIDList())
        self.n_agents = len(self.traffic_light_ids)

        # Initialize phase timers (start with enough time to allow immediate switch)
        self.phase_timers = [self.min_phase_duration] * self.n_agents
        self.switches_this_step = [False] * self.n_agents

        # Run one step to initialize
        traci.simulationStep()
        self.time += self.step_length

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Get state for all agents (matches sumo-marl format).

        State per agent:
        - For each controlled lane: vehicle_count, halting_count
        - Phase one-hot encoding (2 dims)

        Returns:
            State array of shape (n_agents, obs_dim)
        """
        state = []

        for tl_id in self.traffic_light_ids:
            observation = []

            # Get lane counts (vehicle + halting for each controlled lane)
            for lane in traci.trafficlight.getControlledLanes(tl_id):
                observation.append(traci.lane.getLastStepVehicleNumber(lane))
                observation.append(traci.lane.getLastStepHaltingNumber(lane))

            # Phase one-hot encoding (2 phases like sumo-marl)
            phase = [0, 0]
            current_phase = traci.trafficlight.getPhase(tl_id)
            phase[current_phase % self.n_phase] = 1
            observation.extend(phase)

            state.append(np.array(observation, dtype=np.float32))

        # Update obs_dim based on actual observation size
        if state:
            self.obs_dim = len(state[0])

        return np.array(state)

    def _apply_actions(self, actions: list):
        """
        Apply actions to traffic lights with minimum phase duration constraint.

        Args:
            actions: List of actions (0 or 1) for each traffic light
        """
        self.switches_this_step = [False] * self.n_agents

        for i, tl_id in enumerate(self.traffic_light_ids):
            current_phase = traci.trafficlight.getPhase(tl_id)

            # Check if agent wants to switch
            if actions[i] != current_phase:
                # Only allow switch if min_phase_duration has passed
                if self.phase_timers[i] >= self.min_phase_duration:
                    traci.trafficlight.setPhase(tl_id, actions[i])
                    self.phase_timers[i] = 0  # Reset timer
                    self.switches_this_step[i] = True
                else:
                    # Switch blocked, still increment timer
                    self.phase_timers[i] += self.decision_interval
            else:
                # No switch requested, increment timer
                self.phase_timers[i] += self.decision_interval

    def step(self, actions: list) -> tuple:
        """
        Execute one environment step.

        Args:
            actions: List of actions for each agent (0 or 1)

        Returns:
            Tuple of (next_state, rewards, done)
        """
        # Apply actions
        self._apply_actions(actions)

        # Run simulation for decision interval
        steps_to_run = int(self.decision_interval / self.step_length)
        for _ in range(steps_to_run):
            traci.simulationStep()
            self.time += self.step_length

        # Update cumulative metrics
        self.cumulative_waiting_time += self._get_total_waiting()
        self.total_throughput += traci.simulation.getArrivedNumber()

        # Get next state
        next_state = self._get_state()

        # Calculate rewards (simple -halting like sumo-marl)
        rewards = self._get_rewards()

        # Check if done (only time limit, no early termination)
        done = self._is_done()

        return next_state, rewards, done

    def _get_rewards(self) -> np.ndarray:
        """
        Calculate density-normalized rewards for all agents.

        Reward = -(halting_count / max(vehicle_count, MIN_VEHICLES)) - switch_penalty

        This bounds rewards to approximately [-1.05, 0] regardless of traffic density,
        providing stable gradients across light and heavy traffic scenarios.

        Returns:
            Array of rewards for each agent
        """
        MIN_VEHICLES = 5  # Prevent division instability at low traffic
        SWITCH_PENALTY_SCALED = 0.05  # Scaled for [-1, 0] reward range

        rewards = []

        for i, tl_id in enumerate(self.traffic_light_ids):
            halting = 0
            total_vehicles = 0

            for lane in traci.trafficlight.getControlledLanes(tl_id):
                halting += traci.lane.getLastStepHaltingNumber(lane)
                total_vehicles += traci.lane.getLastStepVehicleNumber(lane)

            # Density-normalized reward: proportion of vehicles halting
            effective_vehicles = max(total_vehicles, MIN_VEHICLES)
            reward = -halting / effective_vehicles  # Range: [-1, 0]

            # Add scaled switch penalty
            if self.switches_this_step[i]:
                reward -= SWITCH_PENALTY_SCALED

            rewards.append(reward)

        return np.array(rewards, dtype=np.float32)

    def _get_total_waiting(self) -> int:
        """Get total number of waiting vehicles in the network."""
        total = 0
        for tl_id in self.traffic_light_ids:
            for lane in traci.trafficlight.getControlledLanes(tl_id):
                total += traci.lane.getLastStepHaltingNumber(lane)
        return total

    def _is_done(self) -> bool:
        """Check if episode is finished (only time limit)."""
        return self.time >= self.episode_length

    def close(self):
        """Close the SUMO connection."""
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass

    def get_metrics(self) -> dict:
        """
        Get current simulation metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "time": self.time,
            "cumulative_waiting_time": self.cumulative_waiting_time,
            "total_throughput": self.total_throughput,
            "arrived": traci.simulation.getArrivedNumber(),
            "flow_file": self.current_flow_file,
        }


if __name__ == "__main__":
    # Test the environment
    if "SUMO_HOME" not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")

    print("Testing TrafficEnv...")
    env = TrafficEnv(scenario="2x2", mode="binary")

    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Observation dimension: {env.obs_dim}")
    print(f"Number of agents: {env.n_agents}")
    print(f"Flow file: {env.current_flow_file}")

    # Run a few random steps
    for step in range(10):
        actions = [random.randint(0, 1) for _ in range(env.n_agents)]
        next_state, rewards, done = env.step(actions)
        metrics = env.get_metrics()
        print(f"Step {step}: rewards={rewards.mean():.3f}, throughput={metrics['total_throughput']}, done={done}")

        if done:
            break

    env.close()
    print("Test completed!")
