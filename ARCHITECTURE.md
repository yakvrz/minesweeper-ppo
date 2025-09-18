# ARCHITECTURE.md

Minesweeper RL prototype designed to run locally on a single NVIDIA RTX 4080. This document specifies the core architecture and APIs for the environment, models, training loops, and evaluation. It intentionally omits infra/plumbing (CI/CD, containerization, reproducibility hardening) which will come later.

---

## Goals

* Train an agent to play Minesweeper efficiently on standard boards (8×8×10; later 16×16×40, 16×30×99).
* Keep the code minimal and fast: vectorized NumPy env on CPU, PyTorch model and PPO on GPU.
* Provide clean module boundaries and typed APIs for quick iteration.
* Enable an imitation-learning warm start from a rule-based solver.

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
  train_il.py           # Imitation pretrain from rule solver dataset
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
2. `flags` – {0,1} (optional, auto-maintained by the environment’s deduction loop; the agent never issues flag actions)
3. `adjacent_counts_onehot` – 9 channels for values 0..8; *only* active where `revealed=1`, else all-zero.

Optional helper planes (configurable): frontier mask, remaining mines ratio (broadcast), progress scalar (broadcast).

Optional fast-learning helpers (can be toggled later):

* `frontier` – unknown cells adjacent to any revealed number (1/0)
* Broadcast scalars (e.g., `remaining_mines_ratio`, `progress`) added as extra channels by repeating over `(H,W)`.

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
* **Progress shaping**:

  * `+ (progress_scale * new_safe / (H * W))` per step, where `new_safe` is the number of freshly revealed safe cells after the closure cascade.
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

### Overview

A small fully-convolutional network that preserves spatial structure and emits per-cell action logits.

Backbone: 3×3 conv blocks (32→64→64) with normalization; heads: 1×1 conv policy (2 logits per cell, flattened to `2*H*W`) and value (GAP+MLP); optional mine-prob head; see `minesweeper/models.py`.

### Module Signatures

```python
# models.py
import torch, torch.nn as nn

class CNNPolicy(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.GroupNorm(4, 32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.policy_head = nn.Conv2d(64, 1, 1)  # reveal logit per cell
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.mine_head = nn.Conv2d(64, 1, 1)  # optional

    def forward(self, x, return_mine=False):
        f = self.backbone(x)                 # [B, 64, H, W]
        logits_2 = self.policy_head(f)       # [B, 2, H, W]
        B, _, H, W = logits_2.shape
        policy_logits = logits_2.permute(0,2,3,1).reshape(B, 2*H*W)
        value = self.value_head(f).squeeze(-1)
        if return_mine:
            mine_logits = self.mine_head(f)  # [B,1,H,W]
            return policy_logits, value, mine_logits
        return policy_logits, value
```

**Notes**

* Keep it stride-1 to preserve `(H,W)` throughout.
* Mixed precision and `torch.compile()` recommended on 2.1+.
* Optional `ConvLSTM` can be added later if needed, but the board is effectively Markovian.

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

### Stage A — Imitation Learning (optional but recommended)

* Generate dataset:

  * Random boards; collect states where `forced_moves` exist.
  * For each state, pick one action from the forced set at random.
  * Store `(obs, action_mask, action, mine_mask)` if using auxiliary head.
* Train:

  * Loss = Cross-Entropy(policy) + `λ * BCE(mine_head)` (e.g., `λ = 0.1`).
  * Train for a fixed number of steps/epochs (e.g., 1–3 epochs over \~1–5M samples).

### Stage B — PPO Fine-Tune

* Initialize from IL checkpoint (if used).
* Turn on progress shaping, step penalty, terminal rewards.
* Optionally enable early **frontier masking**; relax after N updates.
* Add curriculum: once eval win-rate > threshold on current board, include the next harder board in a mixture schedule.

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

Example `configs/small_8x8_10.yaml`:

```yaml
env:
  H: 8
  W: 8
  mine_count: 10
  guarantee_safe_neighborhood: true
  step_penalty: 0.0001
  progress_scale: 0.6

ppo:
  num_envs: 256
  steps_per_env: 128
  mini_batches: 8
  ppo_epochs: 3
  gamma: 0.995
  gae_lambda: 0.95
  clip_eps: 0.2
  vf_coef: 0.5
  ent_coef: 0.003
  lr: 0.0003
  max_grad_norm: 0.5
  aux_mine_weight: 0.1
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
3. **M2**: Rule-based solver + imitation warm start; immediate bump in early win-rate.
4. **M3**: Curriculum to 16×16×40 and density randomization; tune entropy/lr.
5. **M4**: Optional mine-prob auxiliary head; visualize heatmaps and correlate with true mines.

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

### Imitation

Imitation training samples only forced-move states for CE/BCE; see `train_il.py`.

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
