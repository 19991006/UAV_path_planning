# DA-MAPPO: Multi-UAV 2D Path Planning

Dynamic Assignment MAPPO for multi-agent path planning in a 2D continuous environment with obstacles.
Supports both MLP MAPPO and GNN MAPPO (agent-count generalizable).

## Dependencies

```
pip install numpy torch scipy matplotlib tensorboard
```

## Training

### MLP MAPPO

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

### GNN MAPPO

```bash
# Smoke test
python train.py --use-gnn --total-updates 1 --rollout-steps 8 --ppo-epochs 1 \
  --minibatch-size 4 --num-agents 3 --num-obstacles 0 --max-steps 20 \
  --eval-interval 0 --save-interval 0

# Full GNN training
python train.py --use-gnn --num-agents 5 --num-obstacles 5 --total-updates 1000
```

When `--use-gnn` is set, `--minibatch-size` means graph time steps (not flattened agent samples),
and `--torch-num-threads` defaults to 1 to avoid CPU oversubscription on small GNN batches.

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-agents` | 3 | Number of UAVs (targets always equal) |
| `--num-obstacles` | 20 | Number of circular obstacles |
| `--assigner-name` | `fixed` | `hungarian`, `greedy`, `fixed`, `cross`, `cbba` |
| `--total-updates` | 1000 | Number of PPO updates |
| `--rollout-steps` | 512 | Environment steps per rollout |
| `--seed` | 42 | Random seed |
| `--run-name` | `mappo` | Identifier for the run |
| `--save-dir` | `runs` | Parent directory for outputs |
| `--use-gnn` | off | Use GNN MAPPO agent (agent-count generalizable) |
| `--layout-mode` | `same_side` | Agent/target layout: `same_side` or `cross` |

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

### Cross-N evaluation (GNN)

GNN checkpoints are agent-count independent — train at one N, evaluate at another:

```bash
python evaluate.py runs/<your_run_dir> --checkpoint best.pt --eval-num-agents 10 --episodes 10
```

All environment and network settings are read automatically from the run's `config.json`.
The best checkpoint (`checkpoints/best.pt`) is loaded automatically.

## GNN architecture

| File | Purpose |
|------|---------|
| `gnn_actor_critic.py` | Graph actor, graph critic, native PyTorch message passing |
| `graph_rollout_buffer.py` | Fixed-N graph rollout buffer |
| `gnn_mappo.py` | MAPPO trainer using graph observations |

The first GNN version uses a directed fully connected agent graph:
- `node_dim = lidar_num_rays + 4 + 2` — `[lidar, ego_motion, assigned_target]`, independent of `num_agents`
- `edge_dim = 4` — `[dx, dy, distance, relative_bearing]`
- Teammate information is represented as edge features so the policy can be reused for different numbers of agents
