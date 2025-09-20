from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import torch


class RolloutBuffer:
    def __init__(
        self,
        num_envs: int,
        steps: int,
        obs_shape: Tuple[int, int, int],
        action_dim: int,
        device: torch.device,
    ):
        self.num_envs = num_envs
        self.steps = steps
        self.device = device

        B = num_envs * steps
        C, H, W = obs_shape
        self.obs = torch.zeros((B, C, H, W), dtype=torch.float32, device=device)
        self.action_mask = torch.zeros((B, action_dim), dtype=torch.bool, device=device)
        self.actions = torch.zeros((B,), dtype=torch.long, device=device)
        self.logp = torch.zeros((B,), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((B,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((B,), dtype=torch.bool, device=device)
        self.values = torch.zeros((B,), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((B,), dtype=torch.float32, device=device)
        self.returns = torch.zeros((B,), dtype=torch.float32, device=device)
        self.mine_labels = None
        self.mine_valid = None

        self._t = 0

    def add(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
        logp: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        mine_labels: torch.Tensor | None = None,
        mine_valid: torch.Tensor | None = None,
    ) -> None:
        bsz = obs.shape[0]
        s = self._t * bsz
        e = s + bsz
        self.obs[s:e] = obs
        self.action_mask[s:e] = action_mask
        self.actions[s:e] = actions
        self.logp[s:e] = logp
        self.rewards[s:e] = rewards
        self.dones[s:e] = dones
        self.values[s:e] = values
        if mine_labels is not None:
            if self.mine_labels is None:
                self.mine_labels = torch.zeros(
                    (self.num_envs * self.steps, obs.shape[-2], obs.shape[-1]),
                    dtype=torch.float32,
                    device=self.device,
                )
            self.mine_labels[s:e] = mine_labels
            if mine_valid is not None:
                if self.mine_valid is None:
                    self.mine_valid = torch.zeros(
                        (self.num_envs * self.steps, obs.shape[-2], obs.shape[-1]),
                        dtype=torch.bool,
                        device=self.device,
                    )
                self.mine_valid[s:e] = mine_valid
        self._t += 1

    def compute_gae(self, last_values: torch.Tensor, gamma: float = 0.995, lam: float = 0.95) -> None:
        N = self.num_envs
        T = self.steps
        rewards = self.rewards.view(T, N)
        values = self.values.view(T, N)
        dones = self.dones.view(T, N)

        advantages = torch.zeros_like(values)
        last_adv = torch.zeros((N,), dtype=torch.float32, device=self.device)
        for t in reversed(range(T)):
            next_value = last_values if t == T - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t].float()
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_adv = delta + gamma * lam * next_non_terminal * last_adv
            advantages[t] = last_adv
        self.advantages = advantages.reshape(-1)
        self.returns = (self.advantages.view(T, N) + values).reshape(-1)

    def get_minibatches(self, batch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        B = self.obs.shape[0]
        idx = torch.randperm(B, device=self.device)
        for s in range(0, B, batch_size):
            sel = idx[s : s + batch_size]
            batch = {
                "obs": self.obs[sel],
                "action_mask": self.action_mask[sel],
                "actions": self.actions[sel],
                "old_logp": self.logp[sel],
                "rewards": self.rewards[sel],
                "dones": self.dones[sel],
                "values": self.values[sel],
                "advantages": self.advantages[sel],
                "returns": self.returns[sel],
            }
            if self.mine_labels is not None:
                batch["mine_labels"] = self.mine_labels[sel]
                if self.mine_valid is not None:
                    batch["mine_valid"] = self.mine_valid[sel]
            yield type("Batch", (), {k: v for k, v in batch.items()})
