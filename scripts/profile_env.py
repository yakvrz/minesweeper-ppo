from __future__ import annotations

import argparse
import time
import numpy as np

from minesweeper.env import EnvConfig, VecMinesweeper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    vec = VecMinesweeper(args.envs, EnvConfig(), seed=args.seed)
    d = vec.reset()
    acts = np.zeros((args.envs,), dtype=np.int32)

    t0 = time.time()
    for _ in range(args.steps):
        # sample random valid action per env
        mask = d["action_mask"]
        for i in range(args.envs):
            valid = np.flatnonzero(mask[i])
            if len(valid) == 0:
                acts[i] = 0
            else:
                acts[i] = valid[np.random.randint(len(valid))]
        d, r, done, info = vec.step(acts)
    dt = time.time() - t0
    print(f"Ran {args.steps} steps across {args.envs} envs in {dt:.3f}s -> {args.envs*args.steps/dt:.1f} steps/s")


if __name__ == "__main__":
    main()


