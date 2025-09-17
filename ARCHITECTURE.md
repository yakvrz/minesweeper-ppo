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

```python
# env.py
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Dict, Any

@dataclass
class EnvConfig:
    H: int = 8
    W: int = 8
    mine_count: int = 10
    guarantee_safe_neighborhood: bool = True  # avoid 8-neighborhood on first click
    progress_reward: float = 0.01
    win_reward: float = 1.0
    loss_reward: float = -1.0
    step_penalty: float = 1e-4
    invalid_penalty: float = 1e-3
    flag_correct_reward: float = 0.002
    flag_incorrect_reward: float = -0.002
    use_flag_shaping: bool = False

class MinesweeperEnv:
    def __init__(self, cfg: EnvConfig, seed: int):
        ...

    def reset(self) -> Dict[str, Any]:
        """Returns observation dict with fields:
           - obs: np.float32 tensor [C, H, W]
           - action_mask: bool [2*H*W]
           - aux: dict with scalars (remaining_mines_ratio, step, ...), optional
        """

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """Action ∈ [0, 2*H*W).
           0..H*W-1 = reveal at idx; H*W..2*H*W-1 = flag toggle at idx.
           Returns (obs_dict, reward, done, info).
        """

    @property
    def action_space(self) -> int:
        return 2 * self.cfg.H * self.cfg.W

    @property
    def obs_channels(self) -> int:
        # revealed, flags, one-hot 0..8 (9 channels) => 11 channels total initially
        return 11
```

### Vectorized Wrapper

```python
class VecMinesweeper:
    """Batched env for throughput. Keeps N independent env states on CPU."""
    def __init__(self, num_envs: int, cfg: EnvConfig, seed: int):
        ...

    def reset(self) -> Dict[str, np.ndarray]:
        """Stacks observations across envs:
           - obs: [N, C, H, W]
           - action_mask: [N, A]
        """

    def step(self, actions: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray, Dict]:
        """actions: [N] int32, returns batch of obs/reward/done/info."""
```

---

## Observation Encoding

Default channels, shape `(C, H, W)`:

1. `revealed` – {0,1}
2. `flags` – {0,1}
3. `adjacent_counts_onehot` – 9 channels for values 0..8; *only* active where `revealed=1`, else all-zero.

Optional fast-learning helpers (can be toggled later):

* `frontier` – unknown cells adjacent to any revealed number (1/0)
* Broadcast scalars (e.g., `remaining_mines_ratio`, `progress`) added as extra channels by repeating over `(H,W)`.

---

## Action Space & Masking

* Discrete size `A = 2 * H * W`.
* Index mapping:

  * `cell = idx % (H*W)`
  * `r = cell // W`, `c = cell % W`
  * `is_flag = (idx >= H*W)`

**Mask rules** (bool mask length `A`):

* For **reveal** actions: mask out if `revealed[r,c]` is True.
* For **flag** actions: mask out if `revealed[r,c]` is True.
* (Curriculum option) During early training, mask reveal actions to **frontier cells only** to reduce pure guessing. Later, relax.

**Implementation tip**: Convert mask to `-inf` before softmax:

```python
masked_logits = logits.masked_fill(~action_mask, -1e9)
```

---

## Rewards

* **Terminal**:

  * Win: `+1.0`
  * Loss: `-1.0`
* **Progress shaping**:

  * `+k` per *new* safe cell revealed in current step (`k ≈ 0.01`); cascade yields a bigger single-step reward.
* **Step penalty**:

  * `-1e-4` per step.
* **Flag shaping** (optional):

  * `+0.002` correct mine flag; `-0.002` incorrect flag.
* **Invalid actions** (should be rare with masking):

  * `-0.001`, no state change.

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

```python
# rules.py
from typing import List, Tuple, Optional

def forced_moves(state) -> List[Tuple[str, int]]:
    """Returns a list of moves [('reveal'|'flag', flat_idx), ...] that are provably correct.
       Empty list if no forced move exists (i.e., guessing required).
    """
```

**Dataset generation**:

* Roll out random boards; whenever `forced_moves` is non-empty:

  * Capture `(obs, action_mask, one_chosen_action)` for policy CE.
  * Optionally capture `(mine_mask)` for auxiliary BCE on mine-prob head (supervised label).
* Avoid adding **guess states** to the dataset.

---

## Model Architecture (PyTorch)

### Overview

A small fully-convolutional network that preserves spatial structure and emits per-cell action logits.

```
Input [B, C, H, W]
  -> ConvBlocks (3×3, channels: 32→64→64; GroupNorm/LayerNorm + ReLU)
  -> Shared feature map [B, F, H, W]
  Heads:
    - PolicyHead: 1×1 conv → 2 logits per cell → reshape to [B, 2*H*W]
    - ValueHead: global average pool → MLP → scalar V(s)
    - (Optional) MineProbHead: 1×1 conv + sigmoid → [B, 1, H, W]
```

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
        self.policy_head = nn.Conv2d(64, 2, 1)  # (reveal, flag) per cell
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

```python
# buffers.py
class RolloutBuffer:
    def __init__(self, num_envs, steps, obs_shape, action_dim, device):
        ...
    def add(self, obs, actions, logp, rewards, dones, values, masks):
        ...
    def compute_gae(self, last_values, gamma=0.995, lam=0.95):
        ...
    def get_minibatches(self, batch_size):
        ...
```

### Training Step

```python
# ppo.py
def ppo_update(model, optimizer, batch, cfg):
    # Apply mask before log-softmax
    logits, value, mine_logits = model(batch.obs, return_mine=cfg.aux_mine_weight > 0)
    masked_logits = logits.masked_fill(~batch.action_mask, -1e9)

    logp = torch.log_softmax(masked_logits, dim=-1)
    logp_act = logp.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

    ratio = (logp_act - batch.old_logp).exp()
    surr1 = ratio * batch.advantages
    surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * batch.advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_clipped = batch.values + (value - batch.values).clamp(-cfg.clip_eps_v, cfg.clip_eps_v)
    v_loss1 = (value - batch.returns).pow(2)
    v_loss2 = (value_clipped - batch.returns).pow(2)
    value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

    ent = -(torch.softmax(masked_logits, -1) * logp).sum(-1).mean()

    loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * ent

    if cfg.aux_mine_weight > 0:
        # BCE with hidden mine mask labels (broadcast to batch)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            mine_logits.squeeze(1), batch.mine_labels)
        loss = loss + cfg.aux_mine_weight * bce

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
    optimizer.step()

    return {
      "loss": float(loss.item()),
      "policy_loss": float(policy_loss.item()),
      "value_loss": float(value_loss.item()),
      "entropy": float(ent.item())
    }
```

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
* Reward components breakdown
* Baselines:

  * Random valid reveal
  * Rule-based solver (no guesses)
  * Heuristic guesser: minimal local mine-prob based on simple constraints

### Eval API

```python
# eval.py
def evaluate(agent, env_cfg, episodes: int = 1000, seed: int = 0) -> Dict[str, float]:
    """Returns dict with win_rate, avg_progress, steps_to_outcome, etc."""
```

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
  progress_reward: 0.01
  win_reward: 1.0
  loss_reward: -1.0
  step_penalty: 0.0001
  invalid_penalty: 0.001
  use_flag_shaping: false

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
  frontier_mask_until_updates: 200  # curriculum switch
```

---

## Edge Cases & Correctness

* **First click safety**: mines placed after first reveal; optionally exclude 8-neighborhood.
* **Flood-fill correctness**: BFS over zero cells; ensure reward counts *unique* newly revealed safe cells.
* **Flags on revealed cells**: invalid (masked + penalty if executed).
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

```python
# train_il.py (sketch)
env = MinesweeperEnv(EnvConfig(), seed=0)
dataset = ForcedMoveDataset(...)  # wraps generator
model = CNNPolicy(in_channels=env.obs_channels).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for batch in loader:
    obs = batch.obs.cuda()                 # [B,C,H,W]
    mask = batch.action_mask.cuda()        # [B,A]
    logits, _, mine_logits = model(obs, return_mine=True)
    logits = logits.masked_fill(~mask, -1e9)
    ce = torch.nn.functional.cross_entropy(logits, batch.action)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        mine_logits.squeeze(1), batch.mine_labels)
    loss = ce + 0.1 * bce
    opt.zero_grad(); loss.backward(); opt.step()
```

### PPO

```python
# train_rl.py (sketch)
vec = VecMinesweeper(num_envs=256, cfg=EnvConfig(), seed=0)
model = CNNPolicy(in_channels=vec.obs_channels()).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for update in range(num_updates):
    roll = collect_rollout(vec, model, steps=128, frontier_mask=update < 200)
    roll.compute_gae(gamma=0.995, lam=0.95)
    for _ in range(3):  # ppo_epochs
        for batch in roll.get_minibatches(...):
            stats = ppo_update(model, opt, batch, cfg=ppo_cfg)
    if should_increase_difficulty(eval_stats):
        mix_in_next_board()
```

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
