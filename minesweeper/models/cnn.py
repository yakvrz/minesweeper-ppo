from __future__ import annotations

import torch
import torch.nn as nn


class CNNPolicy(nn.Module):
    """Lightweight convolutional policy/value network."""

    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        hidden = int(hidden)
        if hidden <= 0:
            raise ValueError("hidden must be positive")

        features = 64
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(4, 32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.policy_head = nn.Conv2d(features, 1, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.mine_head = nn.Conv2d(features, 1, kernel_size=1)

    def set_gradient_checkpointing(self, enabled: bool) -> None:  # pragma: no cover
        # Placeholder for API compatibility; no gradient checkpointing is used here.
        return None

    def forward(self, x: torch.Tensor, return_mine: bool = False):
        f = self.backbone(x)
        B, _, H, W = f.shape

        logits = self.policy_head(f)
        policy_logits_flat = logits.permute(0, 2, 3, 1).reshape(B, H * W)

        value = self.value_head(f).squeeze(-1)
        if return_mine:
            mine_logits_map = self.mine_head(f)
            return policy_logits_flat, value, mine_logits_map
        return policy_logits_flat, value

    def beta_regularizer(self) -> torch.Tensor:
        return next(self.parameters()).new_zeros(())
