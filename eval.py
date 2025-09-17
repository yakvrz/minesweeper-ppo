from __future__ import annotations

from typing import Dict, Optional, Callable
import argparse
import os
import re
import glob
import json

import numpy as np
import torch

from minesweeper.env import EnvConfig, MinesweeperEnv, VecMinesweeper
from minesweeper.models import CNNPolicy


@torch.no_grad()
def evaluate(
    model: CNNPolicy,
    env_cfg: EnvConfig,
    episodes: int = 1000,
    seed: int = 0,
    reveal_only: bool = False,
) -> Dict[str, float]:
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
            if reveal_only:
                # Force reveal-only actions
                half = mask.shape[-1] // 2
                mask[:, half:] = False
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


@torch.no_grad()
def evaluate_vec(
    model: CNNPolicy,
    env_cfg: EnvConfig,
    episodes: int = 1000,
    seed: int = 0,
    num_envs: int = 256,
    progress_every: int = 0,
    print_fn: Optional[Callable[[str], None]] = None,
    reveal_only: bool = False,
    max_steps_per_episode: int = 512,
    reveal_fallback_every: int = 0,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    vec = VecMinesweeper(num_envs=num_envs, cfg=env_cfg, seed=seed)
    d = vec.reset()

    remaining = episodes
    wins = 0
    total_steps = 0
    total_progress = 0.0
    invalids = 0
    processed = 0
    if print_fn is None:
        def print_fn(msg: str):
            print(msg, flush=True)

    while remaining > 0:
        batch_size = min(num_envs, remaining)
        # run episodes for this batch in lockstep
        finished = 0
        # fresh reset already done; loop until we collect batch_size dones
        # We track per-env whether we have counted its episode; after done, we immediately reset but continue counting new episodes only if needed
        counted = np.zeros((num_envs,), dtype=bool)
        step_counters = np.zeros((num_envs,), dtype=np.int32)
        tick = 0
        last_reported_finished = 0
        while finished < batch_size:
            obs = torch.from_numpy(d["obs"]).to(device=device, dtype=torch.float32)
            mask = torch.from_numpy(d["action_mask"]).to(device=device, dtype=torch.bool)
            if reveal_only or (reveal_fallback_every and (tick % reveal_fallback_every == 0)):
                half = mask.shape[-1] // 2
                mask[:, half:] = False
            logits, _ = model(obs)
            masked_logits = logits.masked_fill(~mask, -1e9)
            actions = masked_logits.argmax(dim=-1).cpu().numpy().astype(np.int32)
            d, rewards, dones, info = vec.step(actions)
            step_counters += 1
            outcomes = info.get("outcome", [None] * num_envs)
            for i in range(num_envs):
                if not counted[i] and dones[i]:
                    outcome = outcomes[i]
                    if outcome == "win":
                        wins += 1
                    total_steps += int(step_counters[i])
                    step_counters[i] = 0
                    counted[i] = True
                    finished += 1
                # force-finish overly long episodes
                if not counted[i] and max_steps_per_episode > 0 and step_counters[i] >= max_steps_per_episode:
                    total_steps += int(step_counters[i])
                    step_counters[i] = 0
                    counted[i] = True
                    finished += 1
            tick += 1
            if progress_every:
                if (finished - last_reported_finished) >= min(progress_every, batch_size):
                    print_fn(f"eval progress: {processed + finished}/{episodes} episodes")
                    last_reported_finished = finished
                elif tick % 50 == 0:
                    print_fn(f"eval progress: {processed + finished}/{episodes} episodes (running)")
        remaining -= batch_size
        processed += batch_size
        if progress_every and processed % progress_every == 0:
            print_fn(f"eval progress: {processed}/{episodes} episodes")

    return {
        "win_rate": wins / episodes,
        "avg_steps": total_steps / max(1, episodes),
        "avg_progress": total_progress / max(1, episodes),
        "invalid_rate": invalids / max(1, total_steps),
    }


def _load_latest_checkpoint(run_dir: str) -> str:
    paths = glob.glob(os.path.join(run_dir, "ckpt_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    def ckpt_num(p: str) -> int:
        m = re.search(r"ckpt_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    return max(paths, key=ckpt_num)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Minesweeper RL checkpoint")
    parser.add_argument("--run_dir", type=str, default=None, help="Directory with ckpt_*.pt files")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to a specific checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--config", type=str, required=True, help="YAML config path (to build EnvConfig)")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--progress", action="store_true", help="Print evaluation progress")
    parser.add_argument("--reveal_only", action="store_true", help="Restrict eval actions to reveals only")
    args = parser.parse_args()

    import yaml
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        if not args.run_dir:
            raise SystemExit("Provide either --ckpt or --run_dir")
        ckpt_path = _load_latest_checkpoint(args.run_dir)
    state = torch.load(ckpt_path, map_location=device)

    # Build model
    # We assume in_channels=11 per spec
    model = CNNPolicy(in_channels=11).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    # Env config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    env_cfg = EnvConfig(**cfg["env"]) if "env" in cfg else EnvConfig(**cfg)

    # Evaluate (vectorized, reveal-only)
    metrics = evaluate_vec(
        model,
        env_cfg,
        episodes=args.episodes,
        seed=0,
        num_envs=min(args.num_envs, args.episodes),
        progress_every=(max(1, args.episodes // 4) if args.progress else 0),
        reveal_only=args.reveal_only,
    )

    summary = {"checkpoint": os.path.basename(ckpt_path), **metrics}
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()


