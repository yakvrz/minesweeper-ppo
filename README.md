# minesweeper-rl (prototype)

Single-GPU Minesweeper RL prototype per `ARCHITECTURE.md`.

## Quickstart

- Install deps: `pip install -r requirements.txt`
- Recommended: set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before training to reduce CUDA memory fragmentation (e.g., `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
- Play locally: `python scripts/play_local.py`
- PPO training (CNN): `python train_rl.py --config configs/small_8x8_10.yaml --out runs/ppo`
- Evaluate latest checkpoint: `PYTHONPATH=. python eval.py --run_dir runs/ppo --config configs/small_8x8_10.yaml --episodes 64 --num_envs 64 --progress`

### Interactive Web UI

You can visualize a trained checkpoint inside an interactive Minesweeper board with the model’s safety probabilities overlaid on each unrevealed cell:

- Install deps (including FastAPI): `pip install -r requirements.txt`
- Point the app at a checkpoint (defaults to `runs/baseline_quick20_u200/ckpt_best.pt`): `export MINESWEEPER_CKPT=/path/to/ckpt.pt`
- Launch the web server: `uvicorn webui.app:app --reload`
- Open http://127.0.0.1:8000 in your browser and start revealing tiles. Toggle the overlay to show/hide the semi-transparent probability shading; when enabled, the color + percentage on each hidden cell reflect the model’s belief that the tile is safe (`100%` = safest).
- Optional smoke test for the API layer: `PYTHONPATH=. python scripts/test_webui.py`

## Reveal-only training (tiny example)

- Tiny PPO: `python train_rl.py --config configs/tiny_6x6_6.yaml --out runs/tiny_reveal --updates 200`
- Optional fast smoke run: `python train_rl.py --config configs/smoke.yaml --out runs/smoke --updates 40`

The environment now auto-applies all provable deductions after each reveal (flags + safe reveals + chord). Rewards are:

- `+1 / -1` on win/loss
- `-step_penalty` each step for mild efficiency pressure

No explicit flag actions or penalties remain—the agent purely learns where to reveal.

### Env config knobs

- `step_penalty` (default `1e-4`)

- Win rate (primary) and average steps from the reveal-only evaluation.
- Average normalized progress (`avg_progress`).
- Optional: compare different reveal policies via offline analysis; `eval.py` now always runs the greedy reveal policy.

## Layout

- `minesweeper/env.py`: NumPy env + vectorized wrapper
- `minesweeper/rules.py`: Forced-move solver
- `minesweeper/models.py`: CNN-based policies with shared builder (`build_model`)
- `minesweeper/buffers.py`: PPO rollout buffer with GAE
- `minesweeper/ppo.py`: PPO update (masked)
- `eval.py`: Evaluation CLI and utilities. Reports win rate, belief calibration, and new avoidability metrics (`forced_guess_rate`, `safe_option_rate`, `safe_option_pick_rate`, etc.) that quantify how often the agent truly has to guess versus skipping provably safe tiles.
- `viz.py`: ASCII/heatmap utilities
- `configs/`: YAML configs (`cnn_residual_6x6x4.yaml`, `cnn_residual_8x8x10.yaml`, `cnn_residual_16x16x40.yaml`)
- `scripts/`: Helper scripts (e.g., `play_local.py` for interactive play)

See `ARCHITECTURE.md` for full details.
