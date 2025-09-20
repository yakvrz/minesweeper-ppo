from __future__ import annotations

from typing import Dict, Optional

import torch.nn as nn

from .cnn import CNNPolicy
from .cnn_residual import CNNResidualPolicy
from .transformer import TransformerPolicy

__all__ = [
    "CNNPolicy",
    "CNNResidualPolicy",
    "TransformerPolicy",
    "build_model",
]


def build_model(
    name: str,
    *,
    obs_shape: tuple[int, int, int],
    env_overrides: Dict[str, bool] | None = None,
    model_cfg: Optional[dict] = None,
) -> nn.Module:
    """Factory for policy/value networks.

    Args:
        name: Registered model name ("cnn" or "transformer").
        obs_shape: (C, H, W) input shape from the environment.
        env_overrides: Flags indicating which optional channels are present.
        model_cfg: Extra keyword arguments to pass to the model constructor.
    """

    env_overrides = env_overrides or {}
    cfg = dict(model_cfg or {})

    if name == "cnn":
        in_channels = obs_shape[0]
        hidden = int(cfg.pop("hidden", 64))
        tie_reveal = bool(cfg.pop("tie_reveal_to_belief", False))
        cascade_gamma = float(cfg.pop("cascade_gamma", 1.0))
        # Ignore any transformer-specific keys silently so configs can be shared.
        return CNNPolicy(
            in_channels=in_channels,
            hidden=hidden,
            tie_reveal_to_belief=tie_reveal,
            cascade_gamma=cascade_gamma,
        )

    if name in {"cnn_residual", "cnn_large"}:
        # Alias cnn_large to cnn_residual so configs can choose either label.
        in_channels = obs_shape[0]
        tie_reveal = bool(cfg.pop("tie_reveal_to_belief", False))
        stem_channels = int(cfg.pop("stem_channels", 128))
        blocks = int(cfg.pop("blocks", 6))
        dropout = float(cfg.pop("dropout", 0.05))
        value_hidden = int(cfg.pop("value_hidden", 256))
        cascade_gamma = float(cfg.pop("cascade_gamma", 1.0))
        return CNNResidualPolicy(
            in_channels=in_channels,
            stem_channels=stem_channels,
            blocks=blocks,
            dropout=dropout,
            value_hidden=value_hidden,
            tie_reveal_to_belief=tie_reveal,
            cascade_gamma=cascade_gamma,
        )

    if name == "transformer":
        H, W = obs_shape[1], obs_shape[2]
        include_flags_channel = bool(env_overrides.get("include_flags_channel", False))
        include_frontier_channel = bool(env_overrides.get("include_frontier_channel", False))
        include_remaining_mines_channel = bool(env_overrides.get("include_remaining_mines_channel", False))
        include_progress_channel = bool(env_overrides.get("include_progress_channel", False))
        transformer_cfg = {
            "include_flags_channel": include_flags_channel,
            "include_frontier_channel": include_frontier_channel,
            "include_remaining_mines_channel": include_remaining_mines_channel,
            "include_progress_channel": include_progress_channel,
        }
        transformer_cfg.update(cfg)
        return TransformerPolicy((H, W), **transformer_cfg)

    raise ValueError(f"Unknown model name: {name}")
