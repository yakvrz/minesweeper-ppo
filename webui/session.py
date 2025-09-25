from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from minesweeper.env import EnvConfig, MinesweeperEnv
from minesweeper.models import build_model


@dataclass
class BoardState:
    rows: int
    cols: int
    mine_count: int
    board_label: str
    total_cells: int
    revealed_count: int
    remaining_hidden: int
    mine_probabilities: List[List[Optional[float]]]
    next_move: Optional[Dict[str, Any]]
    flags: List[List[bool]]
    revealed: List[List[bool]]
    counts: List[List[int]]
    done: bool
    outcome: Optional[str]
    step: int


class MinesweeperSession:
    """Interactive Minesweeper session backed by a single 16×16×40 checkpoint."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._rng = np.random.default_rng(seed)

        self.env_cfg = self._load_env_config()
        self.env = self._init_env()

        self.model = self._load_model().to(self.device)
        self.model.eval()

        self._last_obs: Dict[str, Any] = self.env.reset()
        self._user_flags = np.zeros((self.env.H, self.env.W), dtype=bool)
        self._last_done = False
        self._last_outcome: Optional[str] = None

    def _load_env_config(self) -> EnvConfig:
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        cfg = ckpt.get("cfg", {})
        return EnvConfig(
            H=int(cfg.get("H", 16)),
            W=int(cfg.get("W", 16)),
            mine_count=int(cfg.get("mine_count", 40)),
            guarantee_safe_neighborhood=bool(cfg.get("guarantee_safe_neighborhood", True)),
            step_penalty=float(cfg.get("step_penalty", EnvConfig.step_penalty)),
        )

    def _init_env(self, seed: Optional[int] = None) -> MinesweeperEnv:
        rng_seed = int(seed if seed is not None else self._rng.integers(0, 2**31 - 1))
        return MinesweeperEnv(self.env_cfg, seed=rng_seed)

    def _load_model(self) -> torch.nn.Module:
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        model_meta = ckpt.get("model_meta", {})
        name = model_meta.get("name", "cnn_residual")
        model_cfg = model_meta.get("config", {})
        obs_shape = (self.env.obs_channels, self.env.H, self.env.W)
        model = build_model(name, obs_shape=obs_shape, model_cfg=model_cfg)
        state_dict = ckpt.get("model", {})
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=True)
        return model

    def reset(self, seed: Optional[int] = None) -> BoardState:
        self.env = self._init_env(seed=seed)
        self._last_obs = self.env.reset()
        self._user_flags = np.zeros((self.env.H, self.env.W), dtype=bool)
        self._last_done = False
        self._last_outcome = None
        return self._build_state()

    def toggle_flag(self, row: int, col: int) -> BoardState:
        if not (0 <= row < self.env.H and 0 <= col < self.env.W):
            raise ValueError(f"Cell out of bounds: ({row}, {col})")
        if self._last_done or self.env.revealed[row, col]:
            return self._build_state()
        self._user_flags[row, col] = ~self._user_flags[row, col]
        return self._build_state()

    def click(self, row: int, col: int) -> BoardState:
        if not (0 <= row < self.env.H and 0 <= col < self.env.W):
            raise ValueError(f"Cell out of bounds: ({row}, {col})")
        if self._user_flags[row, col]:
            return self._build_state()

        action = row * self.env.W + col
        obs, _reward, done, info = self.env.step(action)
        self._last_obs = obs
        self._user_flags[row, col] = False
        self._last_done = done
        self._last_outcome = info.get("outcome")
        return self._build_state()

    def current_state(self) -> BoardState:
        return self._build_state()

    def _run_inference(self) -> tuple[np.ndarray, Optional[Dict[str, Any]]]:
        obs = self._last_obs["obs"]
        action_mask_flat = self._last_obs["action_mask"].astype(bool)
        mask_grid = action_mask_flat.reshape(self.env.H, self.env.W)
        mask_grid &= ~self._user_flags
        action_mask_flat = mask_grid.reshape(-1)

        obs_tensor = torch.from_numpy(obs[None]).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            policy_logits, _, mine_logits = self.model(obs_tensor, return_mine=True)

        mine_prob = torch.sigmoid(mine_logits).squeeze(0).squeeze(0).cpu().numpy()
        mine_prob_masked = mine_prob.copy()
        mine_prob_masked[self._user_flags] = np.nan
        mine_prob_masked[~mask_grid] = np.nan

        action_mask_tensor = torch.from_numpy(action_mask_flat).to(policy_logits.device, dtype=torch.bool)
        if not bool(action_mask_tensor.any().item()) or self.env.step_count == 0:
            next_move = None
        else:
            masked_logits = policy_logits[0].clone()
            masked_logits[~action_mask_tensor] = -1e9
            best_action = int(masked_logits.argmax().item())
            row, col = divmod(best_action, self.env.W)
            next_move = {
                "action": best_action,
                "row": row,
                "col": col,
                "logit": float(policy_logits[0, best_action].item()),
                "mine_probability": float(mine_prob[row, col]),
            }

        return mine_prob_masked, next_move

    def _build_state(self) -> BoardState:
        mine_prob_map, next_move = self._run_inference()
        revealed = self.env.revealed
        counts = self.env.adjacent_counts

        mine_prob_serializable: List[List[Optional[float]]] = []
        for r in range(self.env.H):
            row_vals: List[Optional[float]] = []
            for c in range(self.env.W):
                val = mine_prob_map[r, c]
                row_vals.append(float(val) if not np.isnan(val) else None)
            mine_prob_serializable.append(row_vals)

        revealed_count = int(revealed.sum())
        total_cells = int(self.env.H * self.env.W)
        remaining_hidden = max(0, total_cells - revealed_count)

        return BoardState(
            rows=self.env.H,
            cols=self.env.W,
            mine_count=int(self.env.cfg.mine_count),
            board_label=f"{self.env.H}×{self.env.W}",
            total_cells=total_cells,
            revealed_count=revealed_count,
            remaining_hidden=remaining_hidden,
            mine_probabilities=mine_prob_serializable,
            next_move=next_move,
            flags=self._user_flags.astype(bool).tolist(),
            revealed=revealed.astype(bool).tolist(),
            counts=counts.astype(int).tolist(),
            done=bool(self._last_done),
            outcome=self._last_outcome,
            step=int(self.env.step_count),
        )
