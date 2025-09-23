# ARCHITECTURE.md

Minesweeper RL prototype designed to run locally on a single NVIDIA RTX 4080. This document specifies the core architecture and APIs for the environment, models, training loops, and evaluation. It intentionally omits infra/plumbing (CI/CD, containerization, reproducibility hardening) which will come later.

---

## Goals

* Train an agent to play Minesweeper efficiently on standard boards (8×8×10; later 16×16×40, 16×30×99).
* Keep the code minimal and fast: vectorized NumPy env on CPU, PyTorch model and PPO on GPU.
* Provide clean module boundaries and typed APIs for quick iteration.

## Non-Goals (for now)

* External game integration/UI automation.
* Distributed training, multi-GPU.
* Full experiment tracking stack; use simple CSV/JSON logs first.

---

## Repo Layout

```
minesweeper/
  env.py                # MinesweeperEnv + VecEnv (NumPy, vectorized)
  rules.py              # Constraint-based rule solver (forced moves only)
  models.py             # CNN policy/value/(optional) mine-prob heads
  buffers.py            # Rollout buffers (PPO), action masks
  ppo.py                # PPO algorithm (masked actions), GAE
  train_rl.py           # PPO training with curriculum + eval
  eval.py               # Eval loop, metrics, win-rate curves
  viz.py                # ASCII / minimal viewer + heatmaps
  configs/
    base.yaml
    small_8x8_10.yaml
    med_16x16_40.yaml
    expert_16x30_99.yaml
  scripts/
    play_local.py       # Human/agent stepping for sanity-check
    profile_env.py      # Microbench of vectorized env
  ARCHITECTURE.md
  README.md
```

---

## Core Concepts

### Board Model

Each environment instance maintains:

* `mine_mask: (H, W) bool` – hidden mines
* `adjacent_counts: (H, W) uint8` – 0–8 numbers
* `revealed: (H, W) bool`
* `flags: (H, W) bool`
* `first_click_done: bool` – mines are placed **after** first reveal to guarantee safety (optionally also clear its 8-neighborhood)
* `rng: np.random.Generator` – per-env RNG

**Zero flood-fill**: Upon revealing a zero cell, BFS cascade reveals its connected zero region and border numbers.

### Episodes

* **Win**: all safe cells revealed
* **Loss**: a mine revealed
* **Step penalties/awards**: see Reward section

---

## Environment API

Environment exposes a dataclass `EnvConfig` and a class `MinesweeperEnv` with `reset()`/`step()` methods returning observation dicts including an action mask; see `minesweeper/env.py`.

### Vectorized Wrapper

Vectorized wrapper `VecMinesweeper` batches N environments on CPU and provides `reset()` and `step()`; see `minesweeper/env.py`.

---

## Observation Encoding

Default channels, shape `(C, H, W)`:

1. `revealed` – {0,1}
2. `adjacent_counts_onehot` – 9 channels for values 0..8; *only* active where `revealed=1`, else all-zero.

Optional helper plane (configurable): broadcast `progress` scalar repeated over `(H,W)`.

---

## Action Space & Masking

* Discrete size `A = H * W` – **reveal actions only**.
* Index mapping: `cell = idx % (H*W)` with `(r, c)` derived from the flat index.

**Mask rule**: mask out any cell that is already revealed. All remaining unknown cells are legal actions.

**Implementation tip**: convert masked logits to `-inf` before the softmax:

```python
masked_logits = logits.masked_fill(~action_mask, -1e9)
```

---

## Rewards

* **Terminal**:

  * Win: `+1.0`
  * Loss: `-1.0`
* **Step penalty**: small constant subtraction each move (default `1e-4`).
* Flag shaping is removed—the environment manages flags automatically during the closure.

---

## Curriculum & Randomization

* Start on 8×8×10; promote when eval win-rate > threshold (e.g., 25–35%).
* Mix densities in a band (e.g., 12–20%) to avoid overfitting to a fixed mine ratio.
* Randomize first click location uniformly.

---

## Rule-Based Solver (Imitation Data)

**Purpose**: Generate “forced move” labels to warm-start policy and optional mine-prob head.

**Core rules (classic constraints):**

* **All-safe rule**: If `number(revealed cell) == #adjacent_flags`, all other adjacent unknowns are safe (reveal).
* **All-mines rule**: If `number(revealed cell) == #adjacent_unknowns`, all those unknowns are mines (flag).

**API:**

Rule-based solver exposes `forced_moves(state)` returning provably correct reveal/flag moves; see `minesweeper/rules.py`.

**Dataset generation**:

* Roll out random boards; whenever `forced_moves` is non-empty:

  * Capture `(obs, action_mask, one_chosen_action)` for policy CE.
  * Optionally capture `(mine_mask)` for auxiliary BCE on mine-prob head (supervised label).
* Avoid adding **guess states** to the dataset.

---

## Model Architecture (PyTorch)

Models are created via a shared `build_model(...)` factory that currently exposes two convolutional backbones.

### CNNPolicy (baseline)

* 3×3 conv stack (32→64→64) with GroupNorm + ReLU.
* Policy head: 1×1 conv → `[B, 1, H, W]` → flatten to `[B, H*W]` reveal logits.
* Value head: GAP → MLP → scalar per batch.
* Auxiliary mine head: 1×1 conv → `[B, 1, H, W]` for optional BCE supervision.

Use this for lightweight baselines or CPU-friendly experiments.

### CNNResidualPolicy (deeper residual stack)

* Stem: 3×3 conv to `stem_channels`, GroupNorm, ReLU.
* Residual stack: `blocks` repeats of 3×3 conv + GroupNorm + ReLU + optional dropout.
* Policy head: 1×1 conv stack → `[B, 1, H, W]` logits.
* Value head: GAP → MLP (`value_hidden`) → scalar.
* Optional mine-prob head and reveal/belief tying identical to the baseline variant.

This is the default for larger boards; it scales width/depth through config values.

### Model Builder

`minesweeper/models.py` exposes `build_model(name, obs_shape, env_overrides, model_cfg)` which instantiates one of the CNN variants. Architecture metadata is stored inside RL checkpoints so `eval.py` can rebuild the exact network automatically.

Mixed precision and `torch.compile()` remain recommended on PyTorch ≥ 2.1.

---

## PPO with Action Masking

### Rollout & Buffer

`RolloutBuffer` stores trajectories, computes GAE, and yields minibatches; see `minesweeper/buffers.py`.

### Training Step

`ppo_update` applies action masking, clipped objective, value clipping, entropy bonus, and optional mine-head BCE; see `minesweeper/ppo.py`.

### PPO Hyperparameters (initial)

* `num_envs`: **256**
* `steps_per_env`: **128** → 32,768 transitions/update
* `mini_batches`: **8**
* `ppo_epochs`: **3**
* `gamma`: **0.995**, `gae_lambda`: **0.95**
* `clip_eps`: **0.2**, (optional) `clip_eps_v`: **0.2**
* `vf_coef`: **0.5**, `ent_coef`: **0.003**
* `lr`: **3e-4** (AdamW), cosine decay
* `max_grad_norm`: **0.5**
* AMP + `torch.compile()` enabled

---

## Training Pipelines

### PPO

* Collect masked-action rollouts from `VecMinesweeper` (typically 256 envs × 128 steps).
* Apply GAE, PPO clipping, value clipping, entropy bonus, and optional mine-head BCE in `ppo_update`.
* Use curriculum or board-mixing as needed (e.g., advance when eval win-rate crosses a threshold).

---

## Evaluation & Debugging

### Metrics

* **Win-rate** per board size/density
* Mean fraction of safe cells revealed
* Steps to win/loss
* % invalid actions (should → \~0)
* Flag metrics: TP/FP counts, toggle rate, precision (optional)
* Reward components breakdown
* Baselines:

  * Random valid reveal
  * Rule-based solver (no guesses)
  * Heuristic guesser: minimal local mine-prob based on simple constraints

### Eval API

`eval.py` provides `evaluate` (single-env) and `evaluate_vec` (vectorized) evaluators and a CLI.

### Visualization

* ASCII viewer for a single env:

  * Board with revealed numbers, flags
  * Agent’s action distribution for the chosen step (top-k highlights)
  * Optional mine-prob heatmap overlay
* Minimal Matplotlib heatmap in `viz.py`.

---

## Performance Notes (RTX 4080)

* Keep env on CPU (NumPy) but vectorized across **256–1024** envs.
* Ship only `obs` (float32 or even float16) and `action_mask` to GPU.
* Reuse buffers; avoid allocations in hot loops.
* Enable AMP (`torch.autocast(device_type="cuda", dtype=torch.float16)`) and `torch.backends.cudnn.benchmark = True`.
* Consider flattening `(N,H,W)` into a single batch per forward and reconstruct indices for actions.

---

## Configuration

Example `configs/cnn_residual_8x8x10.yaml`:

```yaml
env:
  H: 8
  W: 8
  mine_count: 10
  guarantee_safe_neighborhood: true
  step_penalty: 0.0001

model:
  name: cnn_residual
  stem_channels: 160
  blocks: 8
  dropout: 0.05
  value_hidden: 320

ppo:
  num_envs: 128
  steps_per_env: 128
  mini_batches: 8
  ppo_epochs: 3
  gamma: 0.995
  gae_lambda: 0.95
  clip_eps: 0.2
  vf_coef: 0.5
  ent_coef: 0.002
  ent_coef_min: 0.0010
  ent_decay_updates: 300
  lr: 0.0003
  max_grad_norm: 0.5
  aux_mine_weight: 0.05
  aux_mine_calib_weight: 0.01
  total_updates: 1000

training:
  beta_l2: 0.0002
  early_stop_patience: 40
  aux_mine_warmup_weight: 0.08
  aux_mine_warmup_updates: 30
  aux_mine_final_weight: 0.05
  aux_mine_decay_power: 1.0
  rollout:
    num_envs: 128
    steps_per_env: 128
```

---

## Edge Cases & Correctness

* **First click safety**: mines placed after first reveal; optionally exclude 8-neighborhood.
* **Flood-fill correctness**: BFS over zero cells; ensure reward counts *unique* newly revealed safe cells.
* **Auto flags**: deduction loop marks provable mines internally; policy never issues flag actions.
* **Done transitions**: clamp reward to terminal outcome + last step’s progress reward in same tick (documented behavior).
* **Determinism (later)**: seed PyTorch/NumPy and freeze CuDNN for exactness when needed.

---

## Milestones

1. **M0**: Env + ASCII viewer + random agent; sanity checks and unit tests on flood-fill and win/loss detection.
2. **M1**: PPO with action masking on 8×8×10; show progression in avg revealed % and falling invalid rate.
3. **M2**: Curriculum to 16×16×40 and density randomization; tune entropy/lr.
4. **M3**: Optional mine-prob auxiliary head; visualize heatmaps and correlate with true mines.

---

## Minimal Unit Tests (suggested)

* `test_env_initial_click_safe()`
* `test_flood_fill_counts_new_cells()`
* `test_win_loss_detection()`
* `test_action_mask_consistency()`
* `test_rules_forced_moves_soundness()` (no false positives)
* `test_policy_masking_no_invalid_actions()` (after mask application)

---

## Example Training Skeletons

### PPO

RL training collects masked-action rollouts, runs PPO epochs, and can apply curriculum; see `train_rl.py`.

---

## Future Extensions

* Add exact-inference constraint solver for better supervised targets.
* Framing as **POMDP** with belief propagation (via auxiliary head) or explicit CSP loss.
* Off-policy algorithms (Q-learning with masked actions).
* Self-play style curriculum by sampling harder mine densities as proficiency grows.

---

**Contact points**:

* Environment correctness & speed → `env.py`
* Learning dynamics & hyperparams → `ppo.py`, `models.py`
* Solver reproducibility → `rules.py`

This spec should be sufficient to implement the prototype end-to-end on a single machine with an RTX 4080.
