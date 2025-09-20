from __future__ import annotations

import argparse
import os
import sys
import time
import logging
import json
import csv
import math
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
from minesweeper.models import build_model
from minesweeper.buffers import RolloutBuffer
from minesweeper.ppo import PPOConfig, ppo_update
from eval import evaluate_vec


def _average_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_list:
        return {}
    keys = set().union(*metrics_list)
    avg: dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m and m[k] is not None]
        if vals:
            avg[k] = float(sum(vals) / len(vals))
        else:
            avg[k] = float('nan')
    return avg

def _evaluate_model(
    model: nn.Module,
    env_cfg: EnvConfig,
    *,
    episodes: int,
    num_envs: int,
    seed: int,
    pairs: int = 1,
) -> dict[str, float]:
    metrics_list = []
    for i in range(max(1, pairs)):
        metrics = evaluate_vec(
            model,
            env_cfg,
            episodes=episodes,
            seed=seed + i,
            num_envs=num_envs,
            progress_every=0,
        )
        metrics_list.append(metrics)
    return _average_metrics(metrics_list)


torch.set_float32_matmul_precision("high")


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


def load_config(cfg_path: str | None) -> tuple[PPOTrainConfig, dict, dict, dict]:
    if cfg_path is None:
        return PPOTrainConfig(), {}, {}, {}
    with open(cfg_path, "r") as f:
        data = yaml.safe_load(f)
    env_d = data.get("env", {}) or {}
    ppo_d = data.get("ppo", {}) or {}
    model_d = data.get("model", {}) or {}
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
    extras = {k: v for k, v in data.items() if k not in ("env", "ppo", "model")}
    return cfg, env_d, model_d, extras


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
    aux_mine_weight: float = 0.0,
) -> tuple[RolloutBuffer, Dict]:
    batch = vec.reset()
    obs_np = batch["obs"]
    mask_np = batch["action_mask"]
    invalid_rows = ~mask_np.any(axis=1)
    if invalid_rows.any():
        mask_np[invalid_rows] = True
    num_envs = obs_np.shape[0]
    C, H, W = obs_np.shape[1:]
    action_dim = mask_np.shape[1]

    buffer = RolloutBuffer(num_envs=num_envs, steps=steps, obs_shape=(C, H, W), action_dim=action_dim, device=device)

    need_aux = aux_mine_weight > 0
    mine_labels_np = mine_valid_np = None
    mine_labels_cpu = mine_valid_cpu = None
    mine_labels_gpu = mine_valid_gpu = None
    if need_aux:
        mine_labels_np = np.zeros((num_envs, H, W), dtype=np.float32)
        mine_valid_np = np.zeros((num_envs, H, W), dtype=bool)
        mine_labels_cpu = torch.from_numpy(mine_labels_np)
        mine_valid_cpu = torch.from_numpy(mine_valid_np)
        mine_labels_gpu = torch.zeros((num_envs, H, W), dtype=torch.float32, device=device)
        mine_valid_gpu = torch.zeros((num_envs, H, W), dtype=torch.bool, device=device)

    tensor_bridge_time = 0.0
    mine_label_copy_time = 0.0
    env_step_time = 0.0
    model_forward_time = 0.0

    for _ in range(steps):
        t_bridge = time.perf_counter()
        obs = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
        mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool)
        tensor_bridge_time += time.perf_counter() - t_bridge
        mine_labels_tensor = None
        mine_valid_tensor = None
        if need_aux:
            any_labels = False
            for i, env in enumerate(vec.envs):
                if env.first_click_done:
                    any_labels = True
                    np.copyto(mine_labels_np[i], env.mine_mask, casting="unsafe")
                    np.logical_and(~env.revealed, ~env.flags, out=mine_valid_np[i])
                else:
                    mine_labels_np[i].fill(0.0)
                    mine_valid_np[i].fill(False)
            if any_labels:
                t_labels = time.perf_counter()
                mine_labels_gpu.copy_(mine_labels_cpu, non_blocking=True)
                mine_valid_gpu.copy_(mine_valid_cpu, non_blocking=True)
                mine_label_copy_time += time.perf_counter() - t_labels
                mine_labels_tensor = mine_labels_gpu
                mine_valid_tensor = mine_valid_gpu

        t_model = time.perf_counter()
        if need_aux:
            logits, values, _ = model(obs, return_mine=True)
        else:
            logits, values = model(obs)
        model_forward_time += time.perf_counter() - t_model
        logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        logp = dist.log_prob(actions)

        t_bridge = time.perf_counter()
        actions_np = actions.cpu().numpy().astype(np.int32)
        tensor_bridge_time += time.perf_counter() - t_bridge
        t_env = time.perf_counter()
        batch, rewards_np, dones_np, infos = vec.step(actions_np)
        env_step_time += time.perf_counter() - t_env

        t_bridge = time.perf_counter()
        rewards = torch.from_numpy(rewards_np).to(device=device, dtype=torch.float32)
        dones = torch.from_numpy(dones_np).to(device=device, dtype=torch.bool)
        tensor_bridge_time += time.perf_counter() - t_bridge
        buffer.add(
            obs.detach(),
            mask.detach(),
            actions.detach(),
            logp.detach(),
            rewards,
            dones,
            values.detach(),
            mine_labels=mine_labels_tensor,
            mine_valid=mine_valid_tensor,
        )

        obs_np = batch["obs"]
        mask_np = batch["action_mask"]
        invalid_rows = ~mask_np.any(axis=1)
        if invalid_rows.any():
            mask_np[invalid_rows] = True

    t_bridge = time.perf_counter()
    obs = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool)
    tensor_bridge_time += time.perf_counter() - t_bridge
    with torch.no_grad():
        if aux_mine_weight > 0:
            _, last_values, _ = model(obs, return_mine=True)
        else:
            _, last_values = model(obs)
    timings = {
        "steps": steps,
        "tensor_bridge_total_s": tensor_bridge_time,
        "tensor_bridge_per_step_ms": (tensor_bridge_time / steps) * 1000.0 if steps else 0.0,
        "mine_label_copy_total_s": mine_label_copy_time,
        "mine_label_copy_per_step_ms": (mine_label_copy_time / steps) * 1000.0 if steps else 0.0,
        "env_step_total_s": env_step_time,
        "env_step_per_step_ms": (env_step_time / steps) * 1000.0 if steps else 0.0,
        "model_forward_total_s": model_forward_time,
        "model_forward_per_step_ms": (model_forward_time / steps) * 1000.0 if steps else 0.0,
    }
    return buffer, {"last_values": last_values, "timings": timings}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--model", type=str, default=None, help="Model architecture override (cnn, transformer, ...)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="runs/ppo")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--init_ckpt", type=str, default=None, help="Optional init checkpoint from IL")
    parser.add_argument("--eval_episodes", type=int, default=2048)
    parser.add_argument("--eval_num_envs", type=int, default=64)
    parser.add_argument("--save_every", type=int, default=50, help="Save latest checkpoint every N updates")
    parser.add_argument("--eval_quick_episodes", type=int, default=512, help="Episodes for quick periodic eval to track best")
    parser.add_argument("--quick_eval_pairs", type=int, default=2, help="Quick eval repetitions (averaged)")
    parser.add_argument("--quick_eval_interval", type=int, default=10, help="Run quick evaluation every N updates (0 disables)")
    parser.add_argument("--eval_pairs", type=int, default=1, help="Repeat final evaluation batches (averaged)")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Enable gradient checkpointing for transformer blocks")
    args = parser.parse_args()

    cfg, env_overrides, model_cfg, extra_cfg = load_config(args.config)
    training_opts = extra_cfg.get("training", {}) if isinstance(extra_cfg, dict) else {}
    rollout_opts = training_opts.get("rollout", {}) if isinstance(training_opts, dict) else {}
    beta_l2 = float(training_opts.get("beta_l2", 0.0))

    if isinstance(rollout_opts, dict):
        if "num_envs" in rollout_opts:
            cfg.num_envs = int(rollout_opts["num_envs"])
        if "steps_per_env" in rollout_opts:
            cfg.steps_per_env = int(rollout_opts["steps_per_env"])
    if args.updates is not None:
        cfg.total_updates = int(args.updates)

    os.makedirs(args.out, exist_ok=True)

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
    obs_shape = (in_channels, dummy["obs"].shape[2], dummy["obs"].shape[3])

    model_cfg_local = dict(model_cfg)
    model_name = args.model or model_cfg_local.pop("name", "cnn")
    env_flag_overrides = {
        "include_flags_channel": env_cfg.include_flags_channel,
        "include_frontier_channel": env_cfg.include_frontier_channel,
        "include_remaining_mines_channel": env_cfg.include_remaining_mines_channel,
        "include_progress_channel": env_cfg.include_progress_channel,
    }
    model = build_model(
        model_name,
        obs_shape=obs_shape,
        env_overrides=env_flag_overrides,
        model_cfg=model_cfg_local,
    ).to(device)

    grad_ckpt_enabled = bool(args.grad_checkpoint)
    if not grad_ckpt_enabled and isinstance(training_opts, dict):
        grad_ckpt_enabled = bool(training_opts.get("gradient_checkpointing", False))
    if grad_ckpt_enabled and hasattr(model, "set_gradient_checkpointing"):
        model.set_gradient_checkpointing(True)
        log.info("Enabled gradient checkpointing on model")

    model_meta = {
        "name": model_name,
        "config": dict(model_cfg_local),
        "env_flags": env_flag_overrides,
    }

    log.info(f"Model: {model_name} | params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    use_compile = torch.cuda.is_available() and not grad_ckpt_enabled
    if grad_ckpt_enabled and not use_compile:
        log.info("Skipping torch.compile because gradient checkpointing is active")
    if use_compile:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - best effort
            log.warning(f"torch.compile failed: {exc}")
            use_compile = False

    if args.init_ckpt:
        state = torch.load(args.init_ckpt, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state_dict = state["model"]
            if any(k.startswith("_orig_mod.") for k in state_dict):
                state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
            load_status = model.load_state_dict(state_dict, strict=False)
            miss = list(load_status.missing_keys)
            unexpected = list(load_status.unexpected_keys)
            if miss or unexpected:
                log.info("Loaded init checkpoint from %s (missing=%s unexpected=%s)", args.init_ckpt, miss, unexpected)
            else:
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
        beta_l2=beta_l2,
    )

    progress = tqdm(range(cfg.total_updates), desc="updates", disable=not sys.stdout.isatty())

    def _quick_eval_score(metrics: dict[str, float]) -> float:
        def _safe(value):
            if value is None:
                return float('nan')
            try:
                return float(value)
            except Exception:
                return float('nan')

        win = _safe(metrics.get('win_rate'))
        guesses_ep = _safe(metrics.get('guesses_per_episode'))
        guess_success = _safe(metrics.get('guess_success_rate'))
        auroc = _safe(metrics.get('belief_auroc'))
        score = win
        if math.isfinite(guesses_ep):
            score -= max(0.0, guesses_ep - 1.5) * 0.01
            score += max(0.0, 1.5 - guesses_ep) * 0.005
        if math.isfinite(guess_success):
            score += max(0.0, guess_success - 0.75) * 0.05
        if math.isfinite(auroc):
            score += max(0.0, auroc - 0.93) * 0.02
        return score

    per_update_rows = []
    best_quick_score = float('-inf')
    best_quick_metrics: dict[str, float] | None = None

    for update in progress:
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
        buffer, aux = collect_rollout(
            vec,
            model,
            steps=cfg.steps_per_env,
            device=device,
            aux_mine_weight=cfg.aux_mine_weight,
        )
        buffer.compute_gae(aux["last_values"], gamma=cfg.gamma, lam=cfg.gae_lambda)

        B = cfg.num_envs * cfg.steps_per_env
        minibatch_size = B // cfg.mini_batches
        loss_sum = policy_sum = value_sum = ent_sum = 0.0
        aux_sum = 0.0
        aux_count = 0
        count = 0
        for _ in range(cfg.ppo_epochs):
            for batch in buffer.get_minibatches(minibatch_size):
                stats = ppo_update(model, opt, batch, cfg=ppo_cfg)
                loss_sum += stats["loss"]
                policy_sum += stats["policy_loss"]
                value_sum += stats["value_loss"]
                ent_sum += stats["entropy"]
                count += 1
                if "aux_bce" in stats:
                    aux_sum += stats["aux_bce"]
                    aux_count += 1

        sched.step()
        dt = time.time() - t0
        if count > 0:
            loss_avg = loss_sum / count
            pol_avg = policy_sum / count
            val_avg = value_sum / count
            ent_avg = ent_sum / count
            aux_avg = aux_sum / max(1, aux_count) if aux_count > 0 else float("nan")
        else:
            loss_avg = pol_avg = val_avg = ent_avg = aux_avg = float('nan')

        steps_this_update = cfg.num_envs * cfg.steps_per_env
        aux_str = f" aux={aux_avg:.4f}" if aux_count > 0 else ""
        log.info(
            f"upd {update+1}/{cfg.total_updates} | {dt:.2f}s | steps={steps_this_update} | "
            f"pi={pol_avg:.4f} v={val_avg:.4f} ent={ent_avg:.4f}{aux_str} "
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
            "aux_bce": float(aux_avg) if aux_count > 0 else None,
            "quick_win_rate": None,
            "quick_guesses_per_ep": None,
            "quick_guess_success": None,
            "quick_belief_auroc": None,
            "quick_belief_ece": None,
            "quick_score": None,
        })

        if (update + 1) % max(1, args.save_every) == 0:
            ckpt_latest = os.path.join(args.out, "ckpt_latest.pt")
            payload_latest = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "cfg": dict(cfg.__dict__),
                "model_meta": model_meta,
            }
            torch.save(payload_latest, ckpt_latest)

        quick_episodes = min(args.eval_quick_episodes, args.eval_episodes)
        interval = args.quick_eval_interval if args.quick_eval_interval and args.quick_eval_interval > 0 else None
        should_quick_eval = quick_episodes > 0 and (interval is None or (update + 1) % interval == 0)
        if should_quick_eval:
            try:
                metrics_quick = _evaluate_model(
                    model,
                    env_cfg,
                    episodes=quick_episodes,
                    seed=args.seed * 1000 + (update + 1) * 7,
                    num_envs=min(args.eval_num_envs, max(1, quick_episodes // 8)),
                    pairs=args.quick_eval_pairs,
                )
                win_quick = float(metrics_quick.get("win_rate", 0.0))
                row = per_update_rows[-1]
                row["quick_win_rate"] = win_quick
                row["quick_guesses_per_ep"] = metrics_quick.get("guesses_per_episode")
                row["quick_guess_success"] = metrics_quick.get("guess_success_rate")
                row["quick_belief_auroc"] = metrics_quick.get("belief_auroc")
                row["quick_belief_ece"] = metrics_quick.get("belief_ece")
                score = _quick_eval_score(metrics_quick)
                row["quick_score"] = score
                guesses_ep = row["quick_guesses_per_ep"]
                guess_success = row["quick_guess_success"]
                auroc = row["quick_belief_auroc"]
                log.info(
                    "quick eval upd %d: win_rate=%.3f avg_steps=%.2f guesses/ep=%.2f guess_success=%.2f auroc=%.3f score=%.3f",
                    update + 1,
                    win_quick,
                    metrics_quick.get("avg_steps", float('nan')),
                    float(guesses_ep) if guesses_ep is not None else float('nan'),
                    float(guess_success) if guess_success is not None else float('nan'),
                    float(auroc) if auroc is not None else float('nan'),
                    score,
                )
                if score > best_quick_score:
                    best_quick_score = score
                    best_quick_metrics = metrics_quick
                    ckpt_best = os.path.join(args.out, "ckpt_best.pt")
                    payload_best = {
                        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "cfg": dict(cfg.__dict__),
                        "metric": metrics_quick,
                        "model_meta": model_meta,
                    }
                    torch.save(payload_best, ckpt_best)
            except Exception as exc:  # pragma: no cover - best effort
                log.warning(f"Quick eval failed at update {update+1}: {exc}")

    # Export per-update CSV
    try:
        csv_path = os.path.join(args.out, "train_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_update_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_update_rows)
        log.info("Wrote %s", csv_path)
    except Exception as exc:  # pragma: no cover - logging only
        log.warning("Failed writing train_metrics.csv: %s", exc)

    # Final checkpoint
    ckpt_final = os.path.join(args.out, "ckpt_final.pt")
    payload_final = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": dict(cfg.__dict__),
        "model_meta": model_meta,
    }
    torch.save(payload_final, ckpt_final)
    last_ckpt = ckpt_final

    # End-of-training evaluation
    try:
        eval_eps = int(max(8, args.eval_episodes))
        eval_envs = int(min(args.eval_num_envs, eval_eps))
        metrics_raw = _evaluate_model(
            model,
            env_cfg,
            episodes=eval_eps,
            seed=args.seed * 17,
            num_envs=eval_envs,
            pairs=args.eval_pairs,
        )
        summary = {
            "checkpoint": os.path.basename(last_ckpt) if last_ckpt else None,
            "metrics_raw": metrics_raw,
            "model": model_meta,
            "seed": args.seed,
            "quick_eval_pairs": args.quick_eval_pairs,
            "quick_eval_interval": args.quick_eval_interval,
            "eval_pairs": args.eval_pairs,
            "best_quick_metrics": best_quick_metrics,
            "best_quick_score": best_quick_score,
        }
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f)
        log.info("Wrote summary.json: %s", summary)
    except Exception as exc:  # pragma: no cover - logging only
        log.warning("Final eval failed: %s", exc)


if __name__ == "__main__":
    main()
