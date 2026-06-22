# GNN-MAPPO modification

This project keeps the original MLP MAPPO path and adds a GNN path for fixed-N training with cross-N checkpoint execution.

## Key files

- `env2.py`: adds `get_graph_obs()` and `_build_agent_graph()`.
- `gnn_actor_critic.py`: graph actor, graph critic, and native PyTorch message passing.
- `graph_rollout_buffer.py`: fixed-N graph rollout buffer.
- `gnn_mappo.py`: MAPPO trainer using graph observations.
- `train.py`: adds `--use-gnn` and saves GNN checkpoints.
- `evaluate2.py`: uses `env2`, supports `--eval-num-agents` for cross-N evaluation.

## Smoke test

```bash
python train.py --use-gnn --total-updates 1 --rollout-steps 8 --ppo-epochs 1 \
  --minibatch-size 4 --num-agents 3 --num-obstacles 0 --max-steps 20 \
  --eval-interval 0 --save-interval 0
```

## Train GNN

```bash
python train.py --use-gnn --num-agents 5 --num-obstacles 5 --total-updates 1000
```

## Cross-N evaluation

```bash
python evaluate.py runs/<your_run_dir> --checkpoint best.pt --eval-num-agents 10 --episodes 10
```

Notes:

- The first GNN version uses a directed fully connected agent graph.
- `node_dim = lidar_num_rays + 4 + 2`, independent of `num_agents`.
- `edge_dim = 4`: `[dx, dy, distance, relative_bearing]`.
- `--minibatch-size` means graph time steps for the GNN path, not flattened agent samples.
- `--torch-num-threads` defaults to 1 to avoid CPU oversubscription on small GNN batches.
