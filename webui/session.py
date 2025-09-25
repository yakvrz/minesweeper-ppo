from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from minesweeper.env import EnvConfig, MinesweeperEnv
from minesweeper.models import build_model


BOARD_PRESETS: Dict[str, Dict[str, Any]] = {
    "8x8": {"rows": 8, "cols": 8, "mines": 10, "label": "8×8 · 10 mines"},
    "16x16": {"rows": 16, "cols": 16, "mines": 40, "label": "16×16 · 40 mines"},
}


@dataclass
class BoardState:
    rows: int
    cols: int
    mine_count: int
    preset: str
    preset_label: str
    preset_options: List[Dict[str, str]]
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

        self._current_preset = self._match_preset(self.env_cfg)

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

    def set_board_preset(self, preset: str, seed: Optional[int] = None) -> BoardState:
        preset_key = preset.lower()
        if preset_key not in BOARD_PRESETS:
            raise ValueError(f"Unknown board preset: {preset}")
        preset_cfg = BOARD_PRESETS[preset_key]
        self.env_cfg = replace(
            self.env_cfg,
            H=int(preset_cfg["rows"]),
            W=int(preset_cfg["cols"]),
            mine_count=int(preset_cfg["mines"]),
        )
        self._current_preset = preset_key
        return self.reset(seed=seed)

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

        preset_key = self._current_preset
        preset_label = self._preset_label(preset_key)
        preset_options = self._preset_options()

        return BoardState(
            rows=self.env.H,
            cols=self.env.W,
            mine_count=int(self.env.cfg.mine_count),
            preset=preset_key,
            preset_label=preset_label,
            preset_options=preset_options,
            revealed=revealed.astype(bool).tolist(),
            counts=counts.astype(int).tolist(),
            safe_probabilities=safe_prob_serializable,
            done=bool(self._last_done),
            outcome=self._last_outcome,
            step=int(self.env.step_count),
        )

    def _preset_label(self, preset: str) -> str:
        if preset in BOARD_PRESETS:
            return BOARD_PRESETS[preset]["label"]
        return f"{self.env_cfg.H}×{self.env_cfg.W} · {self.env_cfg.mine_count} mines"

    def _preset_options(self) -> List[Dict[str, str]]:
        options = [
            {"id": key, "label": cfg["label"]}
            for key, cfg in BOARD_PRESETS.items()
        ]
        if self._current_preset not in BOARD_PRESETS:
            options.append({
                "id": self._current_preset,
                "label": self._preset_label(self._current_preset),
            })
        return options

    def _match_preset(self, cfg: EnvConfig) -> str:
        for key, preset in BOARD_PRESETS.items():
            if (
                int(preset["rows"]) == int(cfg.H)
                and int(preset["cols"]) == int(cfg.W)
                and int(preset["mines"]) == int(cfg.mine_count)
            ):
                return key
        return "custom"
