from __future__ import annotations

import torch
import torch.nn as nn


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

    def set_gradient_checkpointing(self, enabled: bool) -> None:  # pragma: no cover
        # Placeholder for API compatibility; no gradient checkpointing is used here.
        return None

    def forward(self, x: torch.Tensor, return_mine: bool = False):
        f = self.stem(x)
        f = self.residual_stack(f)

        B, _, H, W = f.shape
        logits = self.policy_head(f)
        policy_logits_flat = logits.permute(0, 2, 3, 1).reshape(B, H * W)

        value = self.value_head(f).squeeze(-1)
        if return_mine:
            # Detach to keep auxiliary belief gradients separate from the policy trunk.
            mine_logits_map = self.mine_head(f.detach())
            return policy_logits_flat, value, mine_logits_map
        return policy_logits_flat, value

    def beta_regularizer(self) -> torch.Tensor:
        return next(self.parameters()).new_zeros(())
