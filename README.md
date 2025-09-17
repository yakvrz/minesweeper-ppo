# minesweeper-rl (prototype)

Single-GPU Minesweeper RL prototype per `ARCHITECTURE.md`.

## Quickstart

- Install deps: `pip install -r requirements.txt`
- Play locally: `python scripts/play_local.py`
- Imitation learning (optional): `python train_il.py --out runs/il`
- PPO training: `python train_rl.py --config configs/small_8x8_10.yaml --out runs/ppo`
- Evaluate latest checkpoint: `PYTHONPATH=. python eval.py --run_dir runs/ppo --config configs/small_8x8_10.yaml --episodes 64 --num_envs 64 --progress`

## Flag-aware training (tiny example)

- Tiny IL (6×6×6): `python train_il.py --out runs/il_tiny --samples 500000 --batch 256 --aux 0.1 --H 6 --W 6 --mines 6`
- Tiny PPO with flags + shaping: `python train_rl.py --config configs/tiny_6x6_6.yaml --out runs/ppo_tiny_flags --updates 200 --init_ckpt runs/il_tiny/il_final.pt`

The tiny config enables:
- Potential-based flag shaping (`alpha_flag`, `flag_toggle_cost`)
- Chord (auto-reveal when flags match numbers)
- Early constraint: flags masked to frontier unknowns
- Stagnation cap to end stalled episodes

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
