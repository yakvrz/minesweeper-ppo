from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from minesweeper.env import EnvConfig, MinesweeperEnv
from minesweeper.models import CNNPolicy
from minesweeper.rules import forced_moves


@dataclass
class ILConfig:
    H: int = 8
    W: int = 8
    mine_count: int = 10
    guarantee_safe_neighborhood: bool = True
    lr: float = 3e-4
    batch_size: int = 256
    total_samples: int = 1_000_000
    aux_mine_weight: float = 0.1


def il_samples(env: MinesweeperEnv, seed: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray, int, np.ndarray]]:
    """Yield (obs[C,H,W], action_mask[A], action_idx, mine_mask[H,W]) for states
    with non-empty forced moves. Progress the env by applying the chosen move.
    Guess steps are used to advance the board but are NOT yielded.
    """
    rng = np.random.default_rng(seed)
    d = env.reset()

    while True:
        fm = forced_moves(env)
        if fm:
            act_type, flat_idx = fm[rng.integers(0, len(fm))]
            H, W = env.H, env.W
            action_idx = flat_idx if act_type == "reveal" else flat_idx + H * W
            yield d["obs"], d["action_mask"], int(action_idx), env.mine_mask.astype(np.float32)
            # step the env using the chosen forced move to continue trajectory
            d, r, done, info = env.step(action_idx)
            if done:
                d = env.reset()
        else:
            # No forced move; take a random valid reveal to progress
            mask = d["action_mask"].copy()
            H, W = env.H, env.W
            reveal_mask = mask[: H * W]
            valid = np.flatnonzero(reveal_mask)
            if len(valid) == 0:
                # fallback: any valid action
                valid = np.flatnonzero(mask)
            if len(valid) == 0:
                d = env.reset()
                continue
            a = int(valid[rng.integers(0, len(valid))])
            d, r, done, info = env.step(a)
            if done:
                d = env.reset()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="runs/il")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--aux", type=float, default=None, help="aux mine head weight")
    args = parser.parse_args()

    cfg = ILConfig()
    if args.samples is not None:
        cfg.total_samples = int(args.samples)
    if args.batch is not None:
        cfg.batch_size = int(args.batch)
    if args.lr is not None:
        cfg.lr = float(args.lr)
    if args.aux is not None:
        cfg.aux_mine_weight = float(args.aux)

    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MinesweeperEnv(EnvConfig(H=cfg.H, W=cfg.W, mine_count=cfg.mine_count, guarantee_safe_neighborhood=cfg.guarantee_safe_neighborhood), seed=args.seed)
    in_channels = env.obs_channels
    model = CNNPolicy(in_channels=in_channels).to(device)
    optim = AdamW(model.parameters(), lr=cfg.lr)

    gen = il_samples(env, seed=args.seed)

    C, H, W = 11, cfg.H, cfg.W
    A = 2 * H * W

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    pbar = tqdm(total=cfg.total_samples, desc="IL samples")
    num_seen = 0
    while num_seen < cfg.total_samples:
        # collect one batch
        obs_list = []
        mask_list = []
        act_list = []
        mine_list = []
        while len(obs_list) < cfg.batch_size and num_seen < cfg.total_samples:
            obs_np, mask_np, action_idx, mine_np = next(gen)
            obs_list.append(torch.from_numpy(obs_np))
            mask_list.append(torch.from_numpy(mask_np))
            act_list.append(action_idx)
            mine_list.append(torch.from_numpy(mine_np))
            num_seen += 1
        obs = torch.stack(obs_list).to(device=device, dtype=torch.float32)
        mask = torch.stack(mask_list).to(device=device, dtype=torch.bool)
        action = torch.tensor(act_list, device=device, dtype=torch.long)
        mine_labels = torch.stack(mine_list).to(device=device, dtype=torch.float32)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits, _, mine_logits = model(obs, return_mine=True)
            logits = logits.masked_fill(~mask, -1e9)
            ce = nn.functional.cross_entropy(logits, action)
            bce = nn.functional.binary_cross_entropy_with_logits(mine_logits.squeeze(1), mine_labels)
            loss = ce + cfg.aux_mine_weight * bce

        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        pbar.update(len(obs_list))

        if (num_seen // cfg.batch_size) % 100 == 0:
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, os.path.join(args.out, "il_latest.pt"))

    # final save
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, os.path.join(args.out, "il_final.pt"))


if __name__ == "__main__":
    main()


