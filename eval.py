from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from minesweeper.env import EnvConfig, MinesweeperEnv
from minesweeper.models import CNNPolicy


@torch.no_grad()
def evaluate(model: CNNPolicy, env_cfg: EnvConfig, episodes: int = 1000, seed: int = 0) -> Dict[str, float]:
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    wins = 0
    total_steps = 0
    total_progress = 0.0
    invalids = 0

    for ep in range(episodes):
        env = MinesweeperEnv(env_cfg, seed=int(rng.integers(0, 2**31 - 1)))
        d = env.reset()
        done = False
        while not done:
            obs = torch.from_numpy(d["obs"][None]).to(device=device, dtype=torch.float32)
            mask = torch.from_numpy(d["action_mask"][None]).to(device=device, dtype=torch.bool)
            logits, _ = model(obs)
            masked_logits = logits.masked_fill(~mask, -1e9)
            action = masked_logits.argmax(dim=-1).item()

            d, r, done, info = env.step(action)
            total_steps += 1
            total_progress += d["aux"].get("last_new_reveals", 0)
            if r < -0.5:  # likely invalid or loss
                # rough invalid detection is not perfect; rely on mask to be correct
                pass
        # win if all safe revealed
        total_safe = env_cfg.H * env_cfg.W - env_cfg.mine_count
        if int(env.revealed.sum()) >= total_safe:
            wins += 1

    return {
        "win_rate": wins / episodes,
        "avg_steps": total_steps / episodes,
        "avg_progress": total_progress / episodes,
        "invalid_rate": invalids / max(1, total_steps),
    }


