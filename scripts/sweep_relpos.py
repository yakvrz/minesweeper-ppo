from __future__ import annotations

import argparse
import subprocess
import tempfile
import yaml
from pathlib import Path


def run_variant(
    base_cfg: Path,
    radius: int,
    updates: int,
    out_dir: Path,
    episodes: int,
    num_envs: int,
    quick_eps: int,
) -> int:
    with open(base_cfg, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("model", {})
    cfg["model"]["rel_pos_radius"] = int(radius)
    cfgout = tempfile.NamedTemporaryFile("w", suffix=f"_r{radius}.yaml", delete=False)
    yaml.safe_dump(cfg, cfgout)
    cfgout.flush()

    target_out = out_dir / f"radius_{radius}"
    target_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "train_rl.py",
        "--config",
        cfgout.name,
        "--out",
        str(target_out),
        "--updates",
        str(updates),
        "--eval_episodes",
        str(episodes),
        "--eval_num_envs",
        str(num_envs),
        "--save_every",
        "1000000",
        "--eval_quick_episodes",
        str(quick_eps),
    ]
    print(f"[sweep] radius={radius} -> {target_out}")
    result = subprocess.run(cmd, check=False)
    cfgout.close()
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Transformer relative attention radius")
    parser.add_argument("--config", type=Path, default=Path("configs/transformer_small.yaml"))
    parser.add_argument("--radii", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--out", type=Path, default=Path("runs/relpos_sweep"))
    parser.add_argument("--eval_episodes", type=int, default=48)
    parser.add_argument("--eval_num_envs", type=int, default=24)
    parser.add_argument("--eval_quick_episodes", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for radius in args.radii:
        code = run_variant(
            args.config,
            radius,
            args.updates,
            args.out,
            args.eval_episodes,
            args.eval_num_envs,
            args.eval_quick_episodes,
        )
        if code != 0:
            print(f"[sweep] radius={radius} exited with code {code}")


if __name__ == "__main__":
    main()
