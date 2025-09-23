from __future__ import annotations

from typing import Dict, Optional

import torch.nn as nn

from .cnn import CNNPolicy
from .cnn_residual import CNNResidualPolicy

__all__ = [
    "CNNPolicy",
    "CNNResidualPolicy",
    "build_model",
]


def build_model(
    name: str,
    *,
    obs_shape: tuple[int, int, int],
    env_overrides: Dict[str, bool] | None = None,
    model_cfg: Optional[dict] = None,
) -> nn.Module:
    env_overrides = env_overrides or {}
    cfg = dict(model_cfg or {})

    if name == "cnn":
        in_channels = obs_shape[0]
        hidden = int(cfg.pop("hidden", 64))
        tie_reveal = bool(cfg.pop("tie_reveal_to_belief", False))
        return CNNPolicy(
            in_channels=in_channels,
            hidden=hidden,
            tie_reveal_to_belief=tie_reveal,
        )

    if name in {"cnn_residual", "cnn_large"}:
        in_channels = obs_shape[0]
        tie_reveal = bool(cfg.pop("tie_reveal_to_belief", False))
        stem_channels = int(cfg.pop("stem_channels", 128))
        blocks = int(cfg.pop("blocks", 6))
        dropout = float(cfg.pop("dropout", 0.05))
        value_hidden = int(cfg.pop("value_hidden", 256))
        return CNNResidualPolicy(
            in_channels=in_channels,
            stem_channels=stem_channels,
            blocks=blocks,
            dropout=dropout,
            value_hidden=value_hidden,
            tie_reveal_to_belief=tie_reveal,
        )

    raise ValueError(f"Unknown model name: {name}")
