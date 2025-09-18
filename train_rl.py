from __future__ import annotations

import argparse
import os
import sys
import time
import logging
import json
import csv
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

from minesweeper.env import EnvConfig, VecMinesweeper
from minesweeper.models import CNNPolicy
from minesweeper.buffers import RolloutBuffer
from minesweeper.ppo import PPOConfig, ppo_update
from eval import evaluate_vec


@dataclass
class PPOTrainConfig:
    # env
    H: int = 8
    W: int = 8
    mine_count: int = 10
    guarantee_safe_neighborhood: bool = True

    # ppo
    num_envs: int = 256
    steps_per_env: int = 128
    mini_batches: int = 8
    ppo_epochs: int = 3
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    clip_eps_v: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.003
    ent_coef_min: float = 0.003
    ent_decay_updates: int = 0
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    aux_mine_weight: float = 0.0
    total_updates: int = 1000


def load_config(cfg_path: str | None) -> tuple[PPOTrainConfig, dict]:
    if cfg_path is None:
        return PPOTrainConfig(), {}
    with open(cfg_path, "r") as f:
        data = yaml.safe_load(f)
    env_d = data.get("env", {}) or {}
    ppo_d = data.get("ppo", {}) or {}
    base = PPOTrainConfig()
    cfg = PPOTrainConfig(
        H=env_d.get("H", base.H),
        W=env_d.get("W", base.W),
        mine_count=env_d.get("mine_count", base.mine_count),
        guarantee_safe_neighborhood=env_d.get("guarantee_safe_neighborhood", base.guarantee_safe_neighborhood),
        num_envs=ppo_d.get("num_envs", base.num_envs),
        steps_per_env=ppo_d.get("steps_per_env", base.steps_per_env),
        mini_batches=ppo_d.get("mini_batches", base.mini_batches),
        ppo_epochs=ppo_d.get("ppo_epochs", base.ppo_epochs),
        gamma=ppo_d.get("gamma", base.gamma),
        gae_lambda=ppo_d.get("gae_lambda", base.gae_lambda),
        clip_eps=ppo_d.get("clip_eps", base.clip_eps),
        clip_eps_v=ppo_d.get("clip_eps_v", base.clip_eps_v),
        vf_coef=ppo_d.get("vf_coef", base.vf_coef),
        ent_coef=ppo_d.get("ent_coef", base.ent_coef),
        ent_coef_min=ppo_d.get("ent_coef_min", base.ent_coef_min),
        ent_decay_updates=ppo_d.get("ent_decay_updates", base.ent_decay_updates),
        lr=ppo_d.get("lr", base.lr),
        max_grad_norm=ppo_d.get("max_grad_norm", base.max_grad_norm),
        aux_mine_weight=ppo_d.get("aux_mine_weight", base.aux_mine_weight),
        total_updates=ppo_d.get("total_updates", base.total_updates),
    )
    return cfg, env_d


@torch.no_grad()
def _forward_policy(model: nn.Module, obs: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    logits, value = model(obs)
    masked_logits = logits.masked_fill(~mask, -1e9)
    logp = torch.log_softmax(masked_logits, dim=-1)
    probs = torch.softmax(masked_logits, dim=-1)
    return {"logits": masked_logits, "logp": logp, "probs": probs, "value": value}


def collect_rollout(
    vec: VecMinesweeper,
    model: nn.Module,
    steps: int,
    device: torch.device,
) -> tuple[RolloutBuffer, Dict]:
    batch = vec.reset()
    obs_np = batch["obs"]
    mask_np = batch["action_mask"]
    num_envs = obs_np.shape[0]
    C, H, W = obs_np.shape[1:]
    action_dim = mask_np.shape[1]

    buffer = RolloutBuffer(num_envs=num_envs, steps=steps, obs_shape=(C, H, W), action_dim=action_dim, device=device)

    for _ in range(steps):
        obs = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
        mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool)
        logits, values = model(obs)
        logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        logp = dist.log_prob(actions)

        # no auxiliary mine labels needed in reveal-only setup
        actions_np = actions.cpu().numpy().astype(np.int32)
        batch, rewards_np, dones_np, infos = vec.step(actions_np)

        rewards = torch.from_numpy(rewards_np).to(device=device, dtype=torch.float32)
        dones = torch.from_numpy(dones_np).to(device=device, dtype=torch.bool)
        buffer.add(
            obs.detach(),
            mask.detach(),
            actions.detach(),
            logp.detach(),
            rewards,
            dones,
            values.detach(),
        )

        obs_np = batch["obs"]
        mask_np = batch["action_mask"]

    obs = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool)
    with torch.no_grad():
        _, last_values = model(obs)
    return buffer, {"last_values": last_values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="runs/ppo")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--init_ckpt", type=str, default=None, help="Optional init checkpoint from IL")
    parser.add_argument("--eval_episodes", type=int, default=64)
    parser.add_argument("--eval_num_envs", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=50, help="Save latest checkpoint every N updates")
    parser.add_argument("--eval_quick_episodes", type=int, default=16, help="Episodes for quick periodic eval to track best")
    args = parser.parse_args()

    cfg, env_overrides = load_config(args.config)
    if args.updates is not None:
        cfg.total_updates = int(args.updates)

    os.makedirs(args.out, exist_ok=True)

    # logging setup
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    log = logging.getLogger("train_rl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    env_kwargs = {
        "H": cfg.H,
        "W": cfg.W,
        "mine_count": cfg.mine_count,
        "guarantee_safe_neighborhood": cfg.guarantee_safe_neighborhood,
    }
    env_kwargs.update(env_overrides)
    env_cfg = EnvConfig(**env_kwargs)
    vec = VecMinesweeper(num_envs=cfg.num_envs, cfg=env_cfg, seed=args.seed)

    dummy = vec.reset()
    in_channels = dummy["obs"].shape[1]
    model = CNNPolicy(in_channels=in_channels).to(device)
    if args.init_ckpt:
        state = torch.load(args.init_ckpt, map_location=device)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
            log.info(f"Loaded init checkpoint from {args.init_ckpt}")
    opt = AdamW(model.parameters(), lr=cfg.lr)
    sched = CosineAnnealingLR(opt, T_max=cfg.total_updates)
    ppo_cfg = PPOConfig(
        clip_eps=cfg.clip_eps,
        clip_eps_v=cfg.clip_eps_v,
        vf_coef=cfg.vf_coef,
        ent_coef=cfg.ent_coef,
        aux_mine_weight=cfg.aux_mine_weight,
        max_grad_norm=cfg.max_grad_norm,
    )

    progress = tqdm(range(cfg.total_updates), desc="updates", disable=not sys.stdout.isatty())
    per_update_rows = []
    best_full_win = -1.0
    for update in progress:
        # Entropy decay if configured
        if cfg.ent_decay_updates > 0:
            decay_steps = max(1, int(cfg.ent_decay_updates))
            decay_frac = min(1.0, update / decay_steps)
            current_ent_coef = float(
                cfg.ent_coef + (cfg.ent_coef_min - cfg.ent_coef) * decay_frac
            )
        else:
            current_ent_coef = float(cfg.ent_coef)
        ppo_cfg.ent_coef = current_ent_coef
        t0 = time.time()
        buffer, aux = collect_rollout(vec, model, steps=cfg.steps_per_env, device=device)
        buffer.compute_gae(aux["last_values"], gamma=cfg.gamma, lam=cfg.gae_lambda)

        B = cfg.num_envs * cfg.steps_per_env
        minibatch_size = B // cfg.mini_batches
        loss_sum = policy_sum = value_sum = ent_sum = 0.0
        count = 0
        for _ in range(cfg.ppo_epochs):
            for batch in buffer.get_minibatches(minibatch_size):
                stats = ppo_update(model, opt, batch, cfg=ppo_cfg)
                loss_sum += stats["loss"]; policy_sum += stats["policy_loss"]; value_sum += stats["value_loss"]; ent_sum += stats["entropy"]; count += 1

        sched.step()
        dt = time.time() - t0
        if count > 0:
            loss_avg = loss_sum / count
            pol_avg = policy_sum / count
            val_avg = value_sum / count
            ent_avg = ent_sum / count
        else:
            loss_avg = pol_avg = val_avg = ent_avg = float('nan')

        steps_this_update = cfg.num_envs * cfg.steps_per_env
        log.info(
            f"upd {update+1}/{cfg.total_updates} | {dt:.2f}s | steps={steps_this_update} | "
            f"loss={loss_avg:.4f} pi={pol_avg:.4f} v={val_avg:.4f} ent={ent_avg:.4f} "
            f"ent_coef={current_ent_coef:.4f}"
        )

        per_update_rows.append({
            "update": int(update + 1),
            "seconds": float(dt),
            "steps": int(steps_this_update),
            "loss": float(loss_avg),
            "policy_loss": float(pol_avg),
            "value_loss": float(val_avg),
            "entropy": float(ent_avg),
            "ent_coef": float(current_ent_coef),
        })

        # Save latest periodically (overwrite) and track best by quick full-mode eval
        if (update + 1) % max(1, args.save_every) == 0:
            ckpt_latest = os.path.join(args.out, "ckpt_latest.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_latest)
        try:
            m_quick = evaluate_vec(
                model,
                env_cfg,
                episodes=min(args.eval_quick_episodes, args.eval_episodes),
                seed=0,
                num_envs=min(args.eval_num_envs, args.eval_quick_episodes),
                progress_every=0,
                reveal_only=False,
                max_steps_per_episode=512,
                reveal_fallback_every=25,
            )
            full_win = float(m_quick.get("win_rate", 0.0))
            if full_win > best_full_win:
                best_full_win = full_win
                ckpt_best = os.path.join(args.out, "ckpt_best.pt")
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "metric": m_quick}, ckpt_best)
        except Exception as e:
            log.warning(f"Quick eval failed at update {update+1}: {e}")

    # Export per-update CSV
    try:
        csv_path = os.path.join(args.out, "train_metrics.csv")
        if per_update_rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(per_update_rows[0].keys()))
                writer.writeheader(); writer.writerows(per_update_rows)
            log.info(f"Wrote {csv_path}")
    except Exception as e:
        log.warning(f"Failed to write train_metrics.csv: {e}")

    # Final checkpoint
    ckpt_final = os.path.join(args.out, "ckpt_final.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_final)
    last_ckpt = ckpt_final

    # End-of-training evaluation (fast reveal-only + full with safeguards)
    try:
        eval_eps = int(max(8, args.eval_episodes))
        eval_envs = int(min(args.eval_num_envs, eval_eps))
        # Reveal-only
        m_reveal = evaluate_vec(model, env_cfg, episodes=eval_eps, seed=0, num_envs=eval_envs, progress_every=0, reveal_only=True)
        # Full with caps
        m_full = evaluate_vec(model, env_cfg, episodes=eval_eps, seed=0, num_envs=eval_envs, progress_every=0, reveal_only=False, max_steps_per_episode=512, reveal_fallback_every=25)
        summary = {
            "checkpoint": os.path.basename(last_ckpt) if last_ckpt else None,
            "reveal_only": m_reveal,
            "full": m_full,
        }
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f)
        log.info(f"Wrote summary.json: {summary}")
    except Exception as e:
        log.warning(f"End-of-training eval failed: {e}")


if __name__ == "__main__":
    main()
