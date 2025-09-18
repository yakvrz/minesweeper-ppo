from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Callable, List
import argparse
import os
import re
import glob
import json

import numpy as np
import torch

from minesweeper.env import EnvConfig, MinesweeperEnv, VecMinesweeper
from minesweeper.models import CNNPolicy


def _assert_action_space(mask: torch.Tensor, logits: torch.Tensor, env: MinesweeperEnv) -> None:
    action_space = env.action_space
    reveal_count = env.H * env.W
    assert logits.shape[-1] == action_space, "policy logits/action dim mismatch"
    assert mask.shape[-1] == action_space, "mask/action dim mismatch"
    assert bool(mask[..., :reveal_count].any().item()), "no valid reveals available"
    assert bool(mask.any().item()), "no valid actions available"


@dataclass
class ControllerState:
    cooldown: torch.Tensor
    last_action: Optional[int] = None
    last_state_changed: bool = True


def _init_controller_state(reveal_count: int, device: torch.device | None = None) -> ControllerState:
    cooldown = torch.zeros(reveal_count, dtype=torch.int64, device=device or torch.device("cpu"))
    return ControllerState(cooldown=cooldown)


def _controller_prepare_state(state: ControllerState, reveal_count: int) -> None:
    device = state.cooldown.device if state.cooldown is not None else torch.device("cpu")
    if state.cooldown.numel() != reveal_count:
        state.cooldown = torch.zeros(reveal_count, dtype=torch.int64, device=device)
    else:
        if state.cooldown.numel() > 0:
            state.cooldown = torch.clamp(state.cooldown - 1, min=0)


def _masked_argmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        scored = logits.masked_fill(~mask, float("-inf"))
        idx = scored.argmax(dim=-1)
        if mask[idx].item():
            return idx
    return logits.argmax(dim=-1)


def _controller_select_action(
    logits: torch.Tensor,
    mask: torch.Tensor,
    state: ControllerState,
    mine_logits: Optional[torch.Tensor],
    reveal_count: int,
    cooldown_steps: int,
) -> int:
    _controller_prepare_state(state, reveal_count)
    local_mask = mask.clone()
    if state.cooldown.numel() == reveal_count:
        flag_cool = state.cooldown > 0
        if flag_cool.any():
            local_mask[reveal_count:][flag_cool] = False

    action = _masked_argmax(logits, local_mask)
    if state.last_action is not None and action.item() == state.last_action and not state.last_state_changed:
        local_mask[action] = False
        if local_mask.any():
            action = _masked_argmax(logits, local_mask)

    if not state.last_state_changed and local_mask[:reveal_count].any():
        if mine_logits is not None:
            probs = torch.sigmoid(mine_logits.flatten())
            masked_probs = probs.masked_fill(~local_mask[:reveal_count], float("inf"))
            alt = torch.argmin(masked_probs)
            if torch.isfinite(masked_probs[alt]):
                action = alt
        else:
            reveal_choice = _masked_argmax(logits[:reveal_count], local_mask[:reveal_count])
            action = reveal_choice

    if action.item() < reveal_count and mine_logits is not None:
        reveal_mask = local_mask[:reveal_count]
        if reveal_mask.any():
            probs = torch.sigmoid(mine_logits.flatten())
            masked_probs = probs.masked_fill(~reveal_mask, float("inf"))
            alt = torch.argmin(masked_probs)
            if torch.isfinite(masked_probs[alt]):
                action = alt

    if not local_mask[action].item():
        # fallback to original mask
        action = _masked_argmax(logits, mask)
    return int(action.item())


def _controller_update_state(
    state: ControllerState,
    action: int,
    reveal_count: int,
    new_reveals: int,
    toggles: int,
    cooldown_steps: int,
) -> None:
    if state.cooldown.numel() != reveal_count:
        state.cooldown = torch.zeros(reveal_count, dtype=torch.int64, device=state.cooldown.device)
    state.last_action = action
    state.last_state_changed = bool(new_reveals > 0 or toggles > 0)
    if action >= reveal_count:
        cell = action - reveal_count
        if 0 <= cell < reveal_count:
            state.cooldown[cell] = cooldown_steps + 1

@torch.no_grad()
def debug_eval(
    model: CNNPolicy,
    env_cfg: EnvConfig,
    reveal_only: bool = False,
    max_steps: int = 512,
    use_controller: bool = False,
    controller_cooldown: int = 2,
) -> None:
    device = next(model.parameters()).device
    env = MinesweeperEnv(env_cfg, seed=0)
    data = env.reset()
    H, W = env.H, env.W
    reveal_count = H * W
    state = _init_controller_state(reveal_count)
    print("[debug] starting single-episode probe", flush=True)
    for step in range(max_steps):
        obs = torch.from_numpy(data["obs"][None]).to(device=device, dtype=torch.float32)
        mask = torch.from_numpy(data["action_mask"][None]).to(device=device, dtype=torch.bool)
        if reveal_only:
            half = mask.shape[-1] // 2
            mask[:, half:] = False

        if use_controller:
            logits, _, mine_logits = model(obs, return_mine=True)
        else:
            logits, _ = model(obs)
            mine_logits = None
        _assert_action_space(mask, logits, env)

        mask_flat = mask[0].view(-1).cpu()
        if use_controller:
            mine_slice = mine_logits[0, 0] if mine_logits is not None else None
            action = _controller_select_action(
                logits[0], mask[0], state, mine_slice, reveal_count, controller_cooldown
            )
        else:
            masked_logits = logits.masked_fill(~mask, -1e9)
            action = int(masked_logits.argmax(dim=-1).item())

        valid_reveals = int(mask_flat[:reveal_count].sum().item())
        valid_flags = int(mask_flat[reveal_count:].sum().item())
        picked_type = "reveal" if action < reveal_count else "flag"
        invalid_action = not bool(mask_flat[action].item())

        prev_reveals = int(env.revealed.sum())
        prev_flags = env.flags.copy()
        if env.first_click_done:
            prev_tp = int((env.flags & env.mine_mask).sum())
            prev_fp = int((env.flags & (~env.mine_mask)).sum())
        else:
            prev_tp = prev_fp = 0

        top_reveal_details = []
        if step < 3:
            reveal_logits = logits[0, :reveal_count].detach().cpu()
            topk = torch.topk(reveal_logits, k=min(5, reveal_count))
            for idx in topk.indices.tolist():
                top_reveal_details.append({
                    "idx": idx,
                    "logit": float(reveal_logits[idx].item()),
                    "mask": bool(mask_flat[idx].item()),
                })

        data, reward, done, info = env.step(action)
        tp_now = int((env.flags & env.mine_mask).sum()) if env.first_click_done else 0
        fp_now = int((env.flags & (~env.mine_mask)).sum()) if env.first_click_done else 0
        new_reveals = int(env.revealed.sum()) - prev_reveals
        toggles = int(np.logical_xor(prev_flags, env.flags).sum())

        if use_controller:
            _controller_update_state(
                state,
                action,
                reveal_count,
                new_reveals,
                toggles,
                controller_cooldown,
            )
        else:
            state.last_action = action
            state.last_state_changed = bool(new_reveals > 0 or toggles > 0)

        print(
            json.dumps(
                {
                    "step": step,
                    "action": int(action),
                    "picked_type": picked_type,
                    "valid_reveals": valid_reveals,
                    "valid_flags": valid_flags,
                    "new_reveals": int(new_reveals),
                    "tp_flags_delta": tp_now - prev_tp,
                    "fp_flags_delta": fp_now - prev_fp,
                    "toggles": int(toggles),
                    "invalid_action": bool(invalid_action),
                    "top_reveal_logits": top_reveal_details,
                }
            ),
            flush=True,
        )
        if done:
            print(json.dumps({"outcome": info.get("outcome"), "steps": step + 1}), flush=True)
            break
    else:
        print(json.dumps({"outcome": "timeout", "steps": max_steps}), flush=True)


@torch.no_grad()
def evaluate(
    model: CNNPolicy,
    env_cfg: EnvConfig,
    episodes: int = 1000,
    seed: int = 0,
    reveal_only: bool = False,
    use_controller: bool = False,
    controller_cooldown: int = 2,
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
        reveal_count = env.H * env.W
        state = _init_controller_state(reveal_count)
        done = False
        while not done:
            obs = torch.from_numpy(d["obs"][None]).to(device=device, dtype=torch.float32)
            mask = torch.from_numpy(d["action_mask"][None]).to(device=device, dtype=torch.bool)
            if reveal_only:
                # Force reveal-only actions
                half = mask.shape[-1] // 2
                mask[:, half:] = False
            if use_controller:
                logits, _, mine_logits = model(obs, return_mine=True)
            else:
                logits, _ = model(obs)
                mine_logits = None
            _assert_action_space(mask, logits, env)
            if use_controller:
                mine_slice = mine_logits[0, 0] if mine_logits is not None else None
                action = _controller_select_action(
                    logits[0], mask[0], state, mine_slice, reveal_count, controller_cooldown
                )
            else:
                masked_logits = logits.masked_fill(~mask, -1e9)
                action = int(masked_logits.argmax(dim=-1).item())
            assert bool(mask.view(-1)[action].item()), "selected action masked out"

            prev_flags = env.flags.copy()
            d, r, done, info = env.step(action)
            total_steps += 1
            new_reveals = int(d["aux"].get("last_new_reveals", 0))
            total_progress += new_reveals / float(env_cfg.H * env_cfg.W)
            toggles = int(np.logical_xor(prev_flags, env.flags).sum())
            if use_controller:
                _controller_update_state(
                    state,
                    action,
                    reveal_count,
                    new_reveals,
                    toggles,
                    controller_cooldown,
                )
            else:
                state.last_action = action
                state.last_state_changed = bool(new_reveals > 0 or toggles > 0)
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
    use_controller: bool = False,
    controller_cooldown: int = 2,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    vec = VecMinesweeper(num_envs=num_envs, cfg=env_cfg, seed=seed)
    d = vec.reset()

    reveal_count = env_cfg.H * env_cfg.W
    controller_states: List[ControllerState] = []
    if use_controller:
        controller_states = [_init_controller_state(reveal_count) for _ in range(num_envs)]

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
        finished = 0
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
            if use_controller:
                logits, _, mine_logits = model(obs, return_mine=True)
            else:
                logits, _ = model(obs)
                mine_logits = None
            _assert_action_space(mask, logits, vec.envs[0])
            if use_controller:
                actions_list = []
                for i in range(mask.shape[0]):
                    mine_slice = mine_logits[i, 0] if mine_logits is not None else None
                    action = _controller_select_action(
                        logits[i], mask[i], controller_states[i], mine_slice, reveal_count, controller_cooldown
                    )
                    actions_list.append(action)
                actions = np.asarray(actions_list, dtype=np.int32)
            else:
                masked_logits = logits.masked_fill(~mask, -1e9)
                actions = masked_logits.argmax(dim=-1).cpu().numpy().astype(np.int32)
            mask_flat = mask.view(mask.shape[0], -1)
            picked_mask = mask_flat[torch.arange(mask.shape[0]), torch.from_numpy(actions)]
            assert bool(picked_mask.all().item()), "selected action masked out"
            d, rewards, dones, info = vec.step(actions)
            step_counters += 1
            aux_list = info.get("aux", [])
            outcomes = info.get("outcome", [None] * num_envs)
            for i in range(num_envs):
                aux = aux_list[i] if i < len(aux_list) else {}
                new_reveals = int(aux.get("last_new_reveals", 0))
                toggles = int(aux.get("toggles", 0))
                if not counted[i]:
                    total_progress += new_reveals / float(env_cfg.H * env_cfg.W)
                    if use_controller:
                        _controller_update_state(
                            controller_states[i],
                            int(actions[i]),
                            reveal_count,
                            new_reveals,
                            toggles,
                            controller_cooldown,
                        )
                if not counted[i] and dones[i]:
                    outcome = outcomes[i]
                    if outcome == "win":
                        wins += 1
                    total_steps += int(step_counters[i])
                    step_counters[i] = 0
                    counted[i] = True
                    finished += 1
                    if use_controller:
                        controller_states[i] = _init_controller_state(reveal_count)
                if not counted[i] and max_steps_per_episode > 0 and step_counters[i] >= max_steps_per_episode:
                    total_steps += int(step_counters[i])
                    step_counters[i] = 0
                    counted[i] = True
                    finished += 1
                    if use_controller:
                        controller_states[i] = _init_controller_state(reveal_count)
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
    parser.add_argument("--debug_eval", action="store_true", help="Run single-episode debug probe")
    parser.add_argument("--controller", action="store_true", help="Enable inference controller heuristics")
    parser.add_argument(
        "--controller_cooldown",
        type=int,
        default=2,
        help="Cooldown steps before the same flag can be reconsidered by the controller",
    )
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

    if args.debug_eval:
        debug_eval(
            model,
            env_cfg,
            reveal_only=args.reveal_only,
            use_controller=args.controller,
            controller_cooldown=args.controller_cooldown,
        )
        return

    # Evaluate (vectorized, reveal-only)
    metrics = evaluate_vec(
        model,
        env_cfg,
        episodes=args.episodes,
        seed=0,
        num_envs=min(args.num_envs, args.episodes),
        progress_every=(max(1, args.episodes // 4) if args.progress else 0),
        reveal_only=args.reveal_only,
        use_controller=args.controller,
        controller_cooldown=args.controller_cooldown,
    )

    summary = {"checkpoint": os.path.basename(ckpt_path), **metrics}
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
