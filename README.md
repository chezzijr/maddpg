# MADDPG Traffic Light Optimization

Multi-Agent Deep Deterministic Policy Gradient (MADDPG) implementation for traffic light control using SUMO traffic simulator. Designed for Vietnamese urban traffic patterns.

## Prerequisites

- **Python** >= 3.11
- **SUMO** (Simulation of Urban Mobility) installed with `SUMO_HOME` environment variable set
- **uv** package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))

### Installing SUMO

```bash
# Ubuntu/Debian
sudo add-apt-repository ppa:sumo/stable
sudo apt update
sudo apt install sumo sumo-tools sumo-doc

# Arch Linux
sudo pacman -S sumo

# macOS
brew install sumo
```

Set the environment variable:
```bash
export SUMO_HOME="/usr/share/sumo"  # adjust path as needed
```

### Installing uv

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd maddpg

# Install dependencies
uv sync
```

## Usage

### Training

```bash
# Train on 2x2 grid (4 traffic lights)
uv run python src/train.py --scenario 2x2 --episodes 500

# Train on 3x3 grid (9 traffic lights)
uv run python src/train.py --scenario 3x3 --episodes 500

# Train with SUMO GUI visualization
uv run python src/train.py --scenario 2x2 --episodes 100 --render

# Resume training from checkpoint
uv run python src/train.py --checkpoint results/2x2/checkpoint_ep100.pt

# Use custom config file
uv run python src/train.py --config configs/custom.yaml

# Force CPU/GPU
uv run python src/train.py --scenario 2x2 --device cpu
uv run python src/train.py --scenario 2x2 --device cuda
```

### Evaluation

```bash
# Quick evaluation of trained model
uv run python src/evaluate.py --checkpoint results/2x2/final_model.pt

# Compare RL model against fixed-timing baseline
uv run python src/baseline_evaluation.py --model results/2x2/final_model.pt
```

### Running Tests

```bash
uv run pytest
```

## Configuration

Training hyperparameters are defined in `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scenario` | `2x2` | Grid size (1x1, 2x2, 3x3, 4x4) |
| `n_episodes` | 500 | Number of training episodes |
| `episode_length` | 3600 | Episode duration in seconds |
| `decision_interval` | 10 | Time between agent decisions (seconds) |
| `lr` | 0.001 | Learning rate (actor and critic) |
| `gamma` | 0.99 | Discount factor |
| `batch_size` | 64 | Training batch size |
| `buffer_size` | 100000 | Replay buffer capacity |
| `eps_start` | 1.0 | Initial exploration rate |
| `eps_end` | 0.05 | Final exploration rate |
| `eps_decay_episodes` | 400 | Episodes for epsilon decay |

## Project Structure

```
maddpg/
├── src/
│   ├── train.py              # Main training script
│   ├── maddpg.py             # MADDPG algorithm
│   ├── env.py                # SUMO environment wrapper
│   ├── networks.py           # Actor/Critic networks
│   ├── replay_buffer.py      # Experience replay
│   ├── utils.py              # Config, logging, plotting
│   ├── evaluate.py           # Model evaluation
│   └── baseline_evaluation.py # RL vs baseline comparison
├── configs/
│   └── default.yaml          # Training configuration
├── scenarios/                # SUMO traffic scenarios
│   ├── 1x1/, 2x2/, 3x3/, 4x4/
│   └── {grid}_vietnamese/
│       ├── network.net.xml   # Road network
│       ├── flows_*.rou.xml   # Traffic demand patterns
│       └── simulation.sumocfg
└── results/                  # Training outputs (generated)
    └── {scenario}/
        ├── final_model.pt    # Trained model
        ├── checkpoint_ep*.pt # Periodic checkpoints
        ├── metrics.csv       # Episode metrics
        └── *.png             # Training curves
```

## Available Scenarios

| Scenario | Agents | Description |
|----------|--------|-------------|
| 1x1 | 1 | Single intersection |
| 2x2 | 4 | 2x2 grid, 4 intersections |
| 3x3 | 9 | 3x3 grid, 9 intersections |
| 4x4 | 16 | 4x4 grid, 16 intersections |

Each scenario includes 6 traffic flow variants for training generalization:
- Balanced traffic
- Light traffic
- Heavy traffic
- North-South dominant
- East-West dominant
- Time-varying patterns
