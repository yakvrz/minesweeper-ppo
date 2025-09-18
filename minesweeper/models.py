from __future__ import annotations

import torch
import torch.nn as nn


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
        self.policy_head = nn.Conv2d(64, 1, 1)
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.mine_head = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor, return_mine: bool = False):
        f = self.backbone(x)  # [B,64,H,W]
        logits_1 = self.policy_head(f)  # [B,1,H,W]
        B, _, H, W = logits_1.shape
        policy_logits = logits_1.permute(0, 2, 3, 1).reshape(B, H * W)
        value = self.value_head(f).squeeze(-1)
        if return_mine:
            mine_logits = self.mine_head(f)  # [B,1,H,W]
            return policy_logits, value, mine_logits
        return policy_logits, value

    # Reveal-only policy: no flag bias controls required
