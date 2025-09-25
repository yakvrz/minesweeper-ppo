# minesweeper-rl

Interactive Minesweeper agent for the 16×16 grid with 40 mines.

## Setup
- Install Python 3.11+ and PyTorch 2.1+
- `pip install -r requirements.txt`

## Web UI
- Optional: `export MINESWEEPER_CKPT_16=/path/to/ckpt.pt`
- `uvicorn webui.app:app --reload`
- Browse http://127.0.0.1:8000
  - Left click reveals a tile
  - Right click (or two-finger tap) toggles a flag
  - The highlighted tile shows the model’s next move and mine probability
- Smoke test: `PYTHONPATH=. python scripts/test_webui.py`

## Evaluation
```bash
PYTHONPATH=. python eval.py \
  --ckpt runs/scaling16_medium_u4000/ckpt_final.pt \
  --config configs/scaling/cnn_residual_16x16x40_medium.yaml \
  --episodes 256 --num_envs 64
```

## Training
```bash
python train_rl.py \
  --config configs/scaling/cnn_residual_16x16x40_medium.yaml \
  --out runs/experiment_name
```

## Layout
- `minesweeper/` core environment, model, PPO code
- `webui/` FastAPI service and static client
- `configs/` training and eval configs
- `scripts/` utilities and tests
- `docs/` supplementary notes
- `runs/` sample checkpoints (not required for operation)

Refer to `ARCHITECTURE.md` for implementation details.
