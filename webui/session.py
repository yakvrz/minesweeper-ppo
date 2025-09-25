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
    revealed: List[List[bool]]
    counts: List[List[int]]
    safe_probabilities: List[List[Optional[float]]]
    done: bool
    outcome: Optional[str]
    step: int


class MinesweeperSession:
    """Load a trained checkpoint and expose interactive env state + probabilities."""

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
        self._last_done = False
        self._last_outcome: Optional[str] = None

    def _load_env_config(self) -> EnvConfig:
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        cfg = ckpt.get("cfg", {})
        return EnvConfig(
            H=int(cfg.get("H", 8)),
            W=int(cfg.get("W", 8)),
            mine_count=int(cfg.get("mine_count", 10)),
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
        self._last_done = False
        self._last_outcome = None
        return self._build_state()

    def click(self, row: int, col: int) -> BoardState:
        if not (0 <= row < self.env.H and 0 <= col < self.env.W):
            raise ValueError(f"Cell out of bounds: ({row}, {col})")

        action = row * self.env.W + col
        obs, reward, done, info = self.env.step(action)
        self._last_obs = obs
        self._last_done = done
        self._last_outcome = info.get("outcome")
        return self._build_state()

    def current_state(self) -> BoardState:
        return self._build_state()

    def _compute_safe_probabilities(self) -> np.ndarray:
        obs = self._last_obs["obs"]
        action_mask = self._last_obs["action_mask"].reshape(self.env.H, self.env.W)
        obs_tensor = torch.from_numpy(obs[None]).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            logits, value, mine_logits = self.model(obs_tensor, return_mine=True)
        if mine_logits is None:
            raise RuntimeError("Checkpoint does not include mine-probability head")
        mine_prob = torch.sigmoid(mine_logits).squeeze(0).squeeze(0).cpu().numpy()
        safe_prob = 1.0 - mine_prob
        # Mask out revealed cells so we do not leak finished tiles.
        safe_prob = safe_prob.copy()
        safe_prob[~action_mask] = np.nan
        return safe_prob

    def _build_state(self) -> BoardState:
        safe_prob = self._compute_safe_probabilities()
        revealed = self.env.revealed
        counts = self.env.adjacent_counts

        safe_prob_serializable: List[List[Optional[float]]] = []
        for r in range(self.env.H):
            row_vals: List[Optional[float]] = []
            for c in range(self.env.W):
                sp = float(safe_prob[r, c]) if not np.isnan(safe_prob[r, c]) else None
                row_vals.append(sp)
            safe_prob_serializable.append(row_vals)

        return BoardState(
            rows=self.env.H,
            cols=self.env.W,
            revealed=revealed.astype(bool).tolist(),
            counts=counts.astype(int).tolist(),
            safe_probabilities=safe_prob_serializable,
            done=bool(self._last_done),
            outcome=self._last_outcome,
            step=int(self.env.step_count),
        )
