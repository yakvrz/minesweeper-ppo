# Minesweeper PPO Agent

End-to-end reinforcement learning project that trains a CNN policy/value network to play 16×16 Minesweeper with 40 mines. The repo packages the training code, evaluation utilities, and a browser-based inspector that reveals what the model believes and which move it would take next.

## Overview
- **Board:** 16×16 grid with 40 mines (Intermediate).
- **Policy:** Residual CNN trained with PPO; no supervised labels were used.
- **Belief head:** Separate mine-probability head for telemetry and UI overlays.
- **Evaluation (2048 episodes):**
  - Win rate: **87.2%** (95% CI: 0.857–0.886)
  - Mean steps per game: **41.8**
  - Mine AUROC: **0.946**
  - Mine ECE: **0.065**
## Web UI
Launch the dashboard to inspect a checkpoint and play interactively:
```bash
pip install -r requirements.txt
export MINESWEEPER_CKPT_16=/path/to/ckpt_final.pt  # optional; defaults to repo checkpoint
uvicorn webui.app:app --reload
```
Open http://127.0.0.1:8000 and:
- Left-click to reveal, right-click (or two-finger tap) to toggle a flag.
- The glowing tile shows the policy’s next move and its mine probability.
- Live counters track step number, revealed cells, and remaining hidden tiles.
- `PYTHONPATH=. python scripts/test_webui.py` runs a quick API smoke check.

## Evaluation
```bash
PYTHONPATH=. python eval.py \
  --ckpt runs/scaling16_medium_u4000/ckpt_final.pt \
  --config configs/eval/16x16x40_medium.yaml \
  --episodes 256 --num_envs 64 --progress
```
Outputs win rate, calibration metrics, and guessing statistics.

## Training
```bash
python train_rl.py \
  --config configs/training/16x16x40_medium.yaml \
  --out runs/experiment_name
```
The script manages PPO rollouts, periodic evaluation, and checkpointing. See `ARCHITECTURE.md` for a full breakdown of the environment and training pipeline.

## Repository Layout
- `minesweeper/` – environment, policy/value networks, PPO implementation
- `webui/` – FastAPI backend and static client
- `configs/` – training (`training/`) and evaluation (`eval/`) configs for 16×16×40
- `scripts/` – helper scripts and smoke tests
- `docs/` – supplementary notes (e.g., architecture overview)
- `runs/` – sample checkpoints and metrics (you can remove or replace with your own)
