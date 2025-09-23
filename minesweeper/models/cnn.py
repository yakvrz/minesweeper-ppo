from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNPolicy(nn.Module):
    """Lightweight convolutional policy/value network."""

    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        *,
        tie_reveal_to_belief: bool = False,
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

        self.tie_reveal_to_belief = bool(tie_reveal_to_belief)
        if self.tie_reveal_to_belief:
            self.belief_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.belief_policy_proj = nn.Sequential(
                nn.LayerNorm(features),
                nn.Linear(features, 2),
            )
        else:
            self.belief_pool = None
            self.belief_policy_proj = None

        self._last_beta_reg: Optional[torch.Tensor] = None

    def set_gradient_checkpointing(self, enabled: bool) -> None:  # pragma: no cover
        # Placeholder for API compatibility; no gradient checkpointing is used here.
        return None

    def forward(self, x: torch.Tensor, return_mine: bool = False):
        f = self.backbone(x)
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
