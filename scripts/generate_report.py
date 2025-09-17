from __future__ import annotations

import argparse
import glob
import json
import os
import re

import torch
import yaml

from minesweeper.models import CNNPolicy
from minesweeper.env import EnvConfig, MinesweeperEnv
from eval import evaluate, evaluate_vec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True, help="Training output directory (with ckpt_*.pt)")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--config", type=str, default="configs/small_8x8_10.yaml")
    p.add_argument("--progress_every", type=int, default=0, help="Print progress every N episodes (0=off)")
    args = p.parse_args()

    print(f"[report] start | run_dir={args.run_dir} | episodes={args.episodes} | config={args.config}", flush=True)

    files = glob.glob(os.path.join(args.run_dir, "ckpt_*.pt"))
    if not files:
        raise SystemExit(f"No checkpoints found in {args.run_dir}")
    def ckpt_num(path: str) -> int:
        m = re.search(r"ckpt_(\d+)\.pt$", os.path.basename(path))
        return int(m.group(1)) if m else -1
    last = max(files, key=ckpt_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(last, map_location=device)
    model = CNNPolicy(in_channels=11).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[report] loaded checkpoint: {os.path.basename(last)}", flush=True)

    cfg = yaml.safe_load(open(args.config, "r"))
    env_cfg = EnvConfig(**cfg["env"]) if "env" in cfg else EnvConfig(**cfg)

    if args.progress_every and args.progress_every > 0:
        # Inline evaluation with progress prints
        import numpy as np
        device = next(model.parameters()).device
        rng = np.random.default_rng(0)
        wins = 0
        total_steps = 0
        total_progress = 0.0
        invalids = 0
        for ep in range(1, args.episodes + 1):
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
            total_safe = env_cfg.H * env_cfg.W - env_cfg.mine_count
            if int(env.revealed.sum()) >= total_safe:
                wins += 1
            if (ep % args.progress_every) == 0:
                print(f"[report] eval progress: {ep}/{args.episodes} episodes", flush=True)
        metrics = {
            "win_rate": wins / args.episodes,
            "avg_steps": total_steps / args.episodes,
            "avg_progress": total_progress / args.episodes,
            "invalid_rate": invalids / max(1, total_steps),
        }
    else:
        print("[report] running vectorized evaluation...", flush=True)
        metrics = evaluate_vec(
            model,
            env_cfg,
            episodes=args.episodes,
            seed=0,
            num_envs=min(256, args.episodes),
            progress_every=max(1, args.episodes // 4),
        )
    summary = {"checkpoint": os.path.basename(last), **metrics}

    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, "summary.json"), "w") as f:
        json.dump(summary, f)

    print(f"[report] done | summary saved to {os.path.join(args.run_dir, 'summary.json')}", flush=True)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()


