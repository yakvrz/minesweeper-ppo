# Minesweeper PPO Agent

End-to-end reinforcement learning project that trains a CNN policy/value network to play 16×16 Minesweeper with 40 mines. The repo packages the training code, evaluation utilities, and a browser-based inspector that reveals what the model believes and which move it would take next.

## At a Glance
- **Board:** 16×16 with 40 hidden mines (classic Intermediate difficulty).
- **Model:** Residual CNN policy/value network with an auxiliary mine-probability head.
- **Training:** PPO on ~4M environment steps (16×16×40 curriculum, medium configuration).
- **Evaluation (256 episodes):**
  - Win rate: **84.0 %** (CI ≈ 0.79–0.88)
  - Average steps to terminal: **54.1**
  - Belief AUC: **0.93**
  - Belief ECE: **0.073**

## Model Card
| Field | Value |
| --- | --- |
| **Architecture** | Residual CNN (stem 128 ch, 6 blocks) with separate mine-probability head for diagnostics |
| **Checkpoint** | `runs/scaling16_medium_u4000/ckpt_final.pt` |
| **Observation** | 10-channel tensor (revealed mask + one-hot counts) |
| **Action space** | Reveal cell only (256 logits) |
| **Training** | PPO (γ=0.995, λ=0.95, 192 envs × 64 steps, 4000 updates); no supervised labels are used for the policy |
| **Reward** | +1 / −1 terminal, −1e-4 per move |
| **Evaluation metrics** | Win 84 %, AUROC 0.93, ECE 0.073 |
| **Limitations** | Mine-probability head is for interpretability only; calibrated on 16×16×40 frontier states |

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

## License
MIT (see `LICENSE`).
