# DA-MAPPO: Multi-UAV 2D Path Planning

Dynamic Assignment MAPPO for multi-agent path planning in a 2D continuous environment with obstacles.

## Dependencies

```
pip install numpy torch scipy matplotlib tensorboard
```

## Training

```bash
# Quick test run
python train.py --total-updates 20 --rollout-steps 128 --num-obstacles 5

# Full training
python train.py --total-updates 1000 --rollout-steps 512 --num-obstacles 20

# With cross-path assignment (agents cross each other)
python train.py --num-agents 5 --num-obstacles 20 --assigner-name cross --run-name cross5

# Resume from checkpoint
python train.py --resume-checkpoint runs/<tag>/checkpoints/final.pt
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-agents` | 3 | Number of UAVs (targets always equal) |
| `--num-obstacles` | 20 | Number of circular obstacles |
| `--assigner-name` | `fixed` | `hungarian`, `greedy`, `fixed`, `cross` |
| `--total-updates` | 1000 | Number of PPO updates |
| `--rollout-steps` | 512 | Environment steps per rollout |
| `--seed` | 42 | Random seed |
| `--run-name` | `mappo` | Identifier for the run |
| `--save-dir` | `runs` | Parent directory for outputs |

Run `python train.py --help` for the full list.

**Outputs** (per run, under `runs/<run_name>_N<agents>_O<obstacles>_<assigner>_S<seed>/`):

| Path | Description |
|------|-------------|
| `config.json` | All training parameters |
| `metrics.csv` | Per-update scalar metrics |
| `tensorboard/` | TensorBoard event files |
| `checkpoints/best.pt` | Best eval-return checkpoint |
| `checkpoints/final.pt` | Final update checkpoint |

Monitor training with:
```bash
tensorboard --logdir runs/<tag>/tensorboard
```

## Evaluation

```bash
# Evaluate with default settings
python evaluate.py runs/mappo_N3_O20_fixed_S42

# More episodes, show plots interactively
python evaluate.py runs/mappo_N5_O20_cross_S42 --episodes 20 --show

# Stochastic policy, no plot output
python evaluate.py runs/mappo_N3_O20_fixed_S42 --stochastic --no-plots
```

All environment and network settings are read automatically from the run's `config.json`.
The best checkpoint (`checkpoints/best.pt`) is loaded automatically.
