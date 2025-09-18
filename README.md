# minesweeper-rl (prototype)

Single-GPU Minesweeper RL prototype per `ARCHITECTURE.md`.

## Quickstart

- Install deps: `pip install -r requirements.txt`
- Play locally: `python scripts/play_local.py`
- Imitation learning (optional): `python train_il.py --out runs/il`
- PPO training: `python train_rl.py --config configs/small_8x8_10.yaml --out runs/ppo`
- Evaluate latest checkpoint: `PYTHONPATH=. python eval.py --run_dir runs/ppo --config configs/small_8x8_10.yaml --episodes 64 --num_envs 64 --progress`

## Reveal-only training (tiny example)

- Tiny PPO: `python train_rl.py --config configs/tiny_6x6_6.yaml --out runs/tiny_reveal --updates 200`
- Optional fast smoke run: `python train_rl.py --config configs/smoke.yaml --out runs/smoke --updates 40`

The environment now auto-applies all provable deductions after each reveal (flags + safe reveals + chord). Rewards are:

- `+1 / -1` on win/loss
- `+(new_safe / (H×W))` per step for newly revealed safe cells

No explicit flag actions or penalties remain—the agent purely learns where to reveal.

## Metrics to monitor

- Full-mode win_rate (primary) and average steps.
- Average normalized progress (`avg_progress` in evaluation output).
- Reveal-only win_rate as a sanity check (pure reveal performance).

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
