# minesweeper-rl (prototype)

Single-GPU Minesweeper RL prototype per `ARCHITECTURE.md`.

## Quickstart

- Install deps: `pip install -r requirements.txt`
- Play locally: `python scripts/play_local.py`
- Imitation learning (optional): `python train_il.py --out runs/il`
- PPO training: `python train_rl.py --config configs/small_8x8_10.yaml --out runs/ppo`
- Evaluate latest checkpoint: `PYTHONPATH=. python eval.py --run_dir runs/ppo --config configs/small_8x8_10.yaml --episodes 64 --num_envs 64 --progress`

## Layout

- `minesweeper/env.py`: NumPy env + vectorized wrapper
- `minesweeper/rules.py`: Forced-move solver
- `minesweeper/models.py`: CNN policy/value (+ optional mine head)
- `minesweeper/buffers.py`: PPO rollout buffer with GAE
- `minesweeper/ppo.py`: PPO update (masked)
- `eval.py`: Evaluation CLI and utilities (`evaluate`, `evaluate_vec`)
- `viz.py`: ASCII/heatmap utilities
- `configs/`: YAML configs
- `scripts/`: Helper scripts

See `ARCHITECTURE.md` for full details.
