# minesweeper-rl

Interactive Minesweeper agent trained on 16×16×40 boards, complete with a FastAPI-powered web UI for exploring the model’s beliefs and recommended moves.

## Requirements

- Python 3.11+
- PyTorch 2.1+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Interactive Web UI

Launch the local dashboard with your checkpoint (defaults to `runs/scaling16_medium_u4000/ckpt_final.pt`, override via `MINESWEEPER_CKPT_16`):

```bash
export MINESWEEPER_CKPT_16=/path/to/ckpt.pt  # optional
uvicorn webui.app:app --reload
```

Open http://127.0.0.1:8000 to play:

- Left-click reveals a tile; right-click (or two-finger tap) toggles a flag.
- The glowing tile shows the policy’s next recommended move and displays the model’s mine probability for that cell.
- Live stats track steps, revealed cells, and remaining hidden tiles.

Run the smoke test to verify the API:

```bash
PYTHONPATH=. python scripts/test_webui.py
```

## Evaluation

To evaluate a checkpoint offline, use the provided scaling config:

```bash
PYTHONPATH=. python eval.py \
  --ckpt runs/scaling16_medium_u4000/ckpt_final.pt \
  --config configs/scaling/cnn_residual_16x16x40_medium.yaml \
  --episodes 256 --num_envs 64 --progress
```

## Training

The project includes reference PPO training scripts. For example, to train the residual CNN on 16×16×40 boards:

```bash
python train_rl.py --config configs/scaling/cnn_residual_16x16x40_medium.yaml --out runs/experiment_name
```

Refer to `ARCHITECTURE.md` for details on the environment, model, and training loop design.

## Repository Layout

- `minesweeper/` – Environment, models, and PPO implementation
- `webui/` – FastAPI service and static client
- `configs/` – Training and evaluation configs (16×16×40)
- `scripts/` – Utilities and tests
- `docs/` – Supplemental notes

Enjoy exploring the agent!
