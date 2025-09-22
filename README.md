# minesweeper-rl (prototype)

Single-GPU Minesweeper RL prototype per `ARCHITECTURE.md`.

## Quickstart

- Install deps: `pip install -r requirements.txt`
- Play locally: `python scripts/play_local.py`
- Imitation learning (optional): `python train_il.py --out runs/il`
- PPO training (CNN): `python train_rl.py --config configs/small_8x8_10.yaml --out runs/ppo`
- PPO training (Transformer full): `python train_rl.py --config configs/transformer.yaml --out runs/transformer --model transformer`
- PPO training (Transformer small): `python train_rl.py --config configs/transformer_small.yaml --out runs/transformer_small --model transformer`
- Evaluate latest checkpoint (auto-detects CNN/Transformer): `PYTHONPATH=. python eval.py --run_dir runs/ppo --config configs/small_8x8_10.yaml --episodes 64 --num_envs 64 --progress`

## Reveal-only training (tiny example)

- Tiny PPO: `python train_rl.py --config configs/tiny_6x6_6.yaml --out runs/tiny_reveal --updates 200`
- Optional fast smoke run: `python train_rl.py --config configs/smoke.yaml --out runs/smoke --updates 40`

The environment now auto-applies all provable deductions after each reveal (flags + safe reveals + chord). Rewards are:

- `+1 / -1` on win/loss
- `+(progress_scale * new_safe / (H×W))` per step for newly revealed safe cells
- `-step_penalty` each step for mild efficiency pressure

No explicit flag actions or penalties remain—the agent purely learns where to reveal.

### Env config knobs

- `step_penalty` (default `1e-4`)
- `progress_scale` (default `0.6`)
- Optional observation channel:
  - `include_progress_channel`

## Metrics to monitor

- Win rate (primary) and average steps from the reveal-only evaluation.
- Average normalized progress (`avg_progress`).
- Optional: compare greedy vs controller-assisted reveal policies if you enable the controller flag in `eval.py`.

## Layout

- `minesweeper/env.py`: NumPy env + vectorized wrapper
- `minesweeper/rules.py`: Forced-move solver
- `minesweeper/models.py`: CNN + Transformer policies with shared builder (`build_model`)
- `minesweeper/buffers.py`: PPO rollout buffer with GAE
- `minesweeper/ppo.py`: PPO update (masked)
- `eval.py`: Evaluation CLI and utilities (`evaluate`, `evaluate_vec`)
- `viz.py`: ASCII/heatmap utilities
- `configs/`: YAML configs (`cnn_residual_6x6x4.yaml`, `cnn_residual_8x8x10.yaml`, `cnn_residual_16x16x40.yaml`)
- `scripts/`: Helper scripts (`sweep_relpos.py` to scan relative attention radius)

See `ARCHITECTURE.md` for full details.
