from __future__ import annotations

import argparse
import os, sys

# Ensure repo root is on sys.path when running from scripts/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from minesweeper.env import EnvConfig, MinesweeperEnv
from viz import ascii_from_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--W", type=int, default=8)
    parser.add_argument("--mines", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = MinesweeperEnv(EnvConfig(H=args.H, W=args.W, mine_count=args.mines), seed=args.seed)
    d = env.reset()
    done = False
    print(ascii_from_env(env))
    while True:
        if done:
            print("Episode ended. Resetting.")
            d = env.reset()
            done = False
            print(ascii_from_env(env))
        s = input("Enter action 'r row col' or 'f row col': ")
        s = s.strip().split()
        if len(s) != 3:
            print("Invalid input. Example: r 3 4")
            continue
        kind, r_str, c_str = s
        r = int(r_str)
        c = int(c_str)
        flat = r * env.W + c
        if kind.lower() == "f":
            a = flat + env.H * env.W
        else:
            a = flat
        d, rwd, done, info = env.step(a)
        print(ascii_from_env(env))
        print(f"reward={rwd:.3f} done={done}")


if __name__ == "__main__":
    main()


