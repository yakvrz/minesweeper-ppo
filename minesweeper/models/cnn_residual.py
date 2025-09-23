from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + residual
        out = self.act(out)
        return out


class CNNResidualPolicy(nn.Module):
    """Higher-capacity convolutional Minesweeper policy/value network."""

    def __init__(
        self,
        in_channels: int,
        *,
        stem_channels: int = 128,
        blocks: int = 6,
        dropout: float = 0.05,
        value_hidden: int = 256,
        tie_reveal_to_belief: bool = False,
    ) -> None:
        super().__init__()
        if stem_channels <= 0:
            raise ValueError("stem_channels must be positive")
        if blocks <= 0:
            raise ValueError("blocks must be positive")

        groups = max(1, stem_channels // 16)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, stem_channels),
            nn.ReLU(inplace=True),
        )

        layers = [_ResidualBlock(stem_channels, groups, dropout=dropout) for _ in range(blocks)]
        self.residual_stack = nn.Sequential(*layers)

        self.policy_head = nn.Sequential(
            nn.Conv2d(stem_channels, stem_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels, 1, kernel_size=1),
        )
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(stem_channels, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1),
        )
        self.mine_head = nn.Sequential(
            nn.Conv2d(stem_channels, stem_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels, 1, kernel_size=1),
        )

        self.tie_reveal_to_belief = bool(tie_reveal_to_belief)
        if self.tie_reveal_to_belief:
            self.belief_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.belief_policy_proj = nn.Sequential(
                nn.LayerNorm(stem_channels),
                nn.Linear(stem_channels, 2),
            )
        else:
            self.belief_pool = None
            self.belief_policy_proj = None

        self._last_beta_reg: Optional[torch.Tensor] = None

    def set_gradient_checkpointing(self, enabled: bool) -> None:  # pragma: no cover
        # Placeholder for API compatibility; no gradient checkpointing is used here.
        return None

    def forward(self, x: torch.Tensor, return_mine: bool = False):
        f = self.stem(x)
        f = self.residual_stack(f)

        B, _, H, W = f.shape
        mine_logits_map: Optional[torch.Tensor] = None
        mine_logits_flat: Optional[torch.Tensor] = None
        if return_mine or self.tie_reveal_to_belief:
            mine_logits_map = self.mine_head(f)
            mine_logits_flat = mine_logits_map.view(B, -1)

        if self.tie_reveal_to_belief:
            assert self.belief_pool is not None and self.belief_policy_proj is not None
            pooled = self.belief_pool(f)
            params = self.belief_policy_proj(pooled)
            beta_raw, bias = params.split(1, dim=-1)
            beta = F.softplus(beta_raw) + 1e-3
            self._last_beta_reg = (beta ** 2).mean()
            if mine_logits_flat is None:
                mine_logits_map = self.mine_head(f)
                mine_logits_flat = mine_logits_map.view(B, -1)
            policy_logits_flat = (-beta * mine_logits_flat + bias).view(B, H * W)
        else:
            self._last_beta_reg = None
            logits = self.policy_head(f)
            policy_logits_flat = logits.permute(0, 2, 3, 1).reshape(B, H * W)

        value = self.value_head(f).squeeze(-1)
        if return_mine:
            if mine_logits_map is None:
                mine_logits_map = self.mine_head(f)
            return policy_logits_flat, value, mine_logits_map
        return policy_logits_flat, value

    def beta_regularizer(self) -> torch.Tensor:
        if self._last_beta_reg is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self._last_beta_reg
