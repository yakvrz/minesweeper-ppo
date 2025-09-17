from __future__ import annotations

import argparse
import os
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
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    aux_mine_weight: float = 0.0
    frontier_mask_until_updates: int = 200
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
        lr=ppo_d.get("lr", base.lr),
        max_grad_norm=ppo_d.get("max_grad_norm", base.max_grad_norm),
        aux_mine_weight=ppo_d.get("aux_mine_weight", base.aux_mine_weight),
        frontier_mask_until_updates=ppo_d.get("frontier_mask_until_updates", base.frontier_mask_until_updates),
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


def collect_rollout(vec: VecMinesweeper, model: nn.Module, steps: int, device: torch.device) -> tuple[RolloutBuffer, Dict]:
    vec_batch = vec.reset()
    N = vec_batch["obs"].shape[0]
    C, H, W = vec_batch["obs"].shape[1:]
    A = vec_batch["action_mask"].shape[1]

    buffer = RolloutBuffer(num_envs=N, steps=steps, obs_shape=(C, H, W), action_dim=A, device=device)

    obs_np = vec_batch["obs"]
    mask_np = vec_batch["action_mask"]

    for t in range(steps):
        obs = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
        mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool)

        out = _forward_policy(model, obs, mask)
        probs = out["probs"]
        dist = torch.distributions.Categorical(probs=probs)
        actions = dist.sample()
        logp = dist.log_prob(actions)
        values = out["value"]

        # prepare auxiliary mine labels from current env states (optional)
        mine_labels_np = []
        for e in vec.envs:
            if getattr(e, "first_click_done", False):
                mine_labels_np.append(e.mine_mask.astype(np.float32))
            else:
                mine_labels_np.append(np.zeros((e.H, e.W), dtype=np.float32))
        mine_labels = torch.from_numpy(np.stack(mine_labels_np, axis=0)).to(device=device, dtype=torch.float32)

        # step env
        actions_np = actions.detach().cpu().numpy().astype(np.int32)
        batch, rewards_np, dones_np, infos = vec.step(actions_np)

        # add to buffer
        rewards = torch.from_numpy(rewards_np).to(device=device, dtype=torch.float32)
        dones = torch.from_numpy(dones_np).to(device=device, dtype=torch.bool)
        buffer.add(obs, mask, actions, logp, rewards, dones, values, mine_labels=mine_labels)

        obs_np = batch["obs"]
        mask_np = batch["action_mask"]

    # bootstrap value
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
    args = parser.parse_args()

    cfg, env_overrides = load_config(args.config)
    if args.updates is not None:
        cfg.total_updates = int(args.updates)

    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    env_cfg = EnvConfig(
        H=cfg.H,
        W=cfg.W,
        mine_count=cfg.mine_count,
        guarantee_safe_neighborhood=cfg.guarantee_safe_neighborhood,
        progress_reward=env_overrides.get("progress_reward", EnvConfig().progress_reward),
        win_reward=env_overrides.get("win_reward", EnvConfig().win_reward),
        loss_reward=env_overrides.get("loss_reward", EnvConfig().loss_reward),
        step_penalty=env_overrides.get("step_penalty", EnvConfig().step_penalty),
        invalid_penalty=env_overrides.get("invalid_penalty", EnvConfig().invalid_penalty),
        flag_correct_reward=env_overrides.get("flag_correct_reward", EnvConfig().flag_correct_reward),
        flag_incorrect_reward=env_overrides.get("flag_incorrect_reward", EnvConfig().flag_incorrect_reward),
        use_flag_shaping=env_overrides.get("use_flag_shaping", EnvConfig().use_flag_shaping),
    )
    vec = VecMinesweeper(num_envs=cfg.num_envs, cfg=env_cfg, seed=args.seed)

    dummy = vec.reset()
    in_channels = dummy["obs"].shape[1]
    model = CNNPolicy(in_channels=in_channels).to(device)
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

    for update in tqdm(range(cfg.total_updates), desc="updates"):
        vec.set_frontier_only_reveal(update < cfg.frontier_mask_until_updates)
        buffer, aux = collect_rollout(vec, model, steps=cfg.steps_per_env, device=device)
        buffer.compute_gae(aux["last_values"], gamma=cfg.gamma, lam=cfg.gae_lambda)

        B = cfg.num_envs * cfg.steps_per_env
        minibatch_size = B // cfg.mini_batches
        for _ in range(cfg.ppo_epochs):
            for batch in buffer.get_minibatches(minibatch_size):
                stats = ppo_update(model, opt, batch, cfg=ppo_cfg)

        sched.step()

        if (update + 1) % 10 == 0:
            ckpt_path = os.path.join(args.out, f"ckpt_{update+1}.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)


if __name__ == "__main__":
    main()


