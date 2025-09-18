from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy as np
from collections import deque

from .rules import forced_moves


@dataclass
class EnvConfig:
    H: int = 8
    W: int = 8
    mine_count: int = 10
    guarantee_safe_neighborhood: bool = True

    win_reward: float = 1.0
    loss_reward: float = -1.0


class MinesweeperEnv:
    """Single Minesweeper environment running on CPU with NumPy state.

    Observation encoding (C,H,W):
      0: revealed mask {0,1}
      1: flags mask {0,1}
      2..10: one-hot for adjacent count 0..8 (active only where revealed=1)
    """

    def __init__(self, cfg: EnvConfig, seed: int = 0):
        self.cfg = cfg
        self.H = int(cfg.H)
        self.W = int(cfg.W)
        self.A = self.H * self.W

        self.rng = np.random.default_rng(seed)

        # Runtime state
        self.mine_mask: np.ndarray
        self.adjacent_counts: np.ndarray
        self.revealed: np.ndarray
        self.flags: np.ndarray
        self.first_click_done: bool
        self.step_count: int
        self._last_new_reveals: int = 0

        self.reset()

    # ------------------------ Public API ------------------------
    def reset(self) -> Dict[str, Any]:
        self.mine_mask = np.zeros((self.H, self.W), dtype=bool)
        self.adjacent_counts = np.zeros((self.H, self.W), dtype=np.uint8)
        self.revealed = np.zeros((self.H, self.W), dtype=bool)
        self.flags = np.zeros((self.H, self.W), dtype=bool)
        self.first_click_done = False
        self.step_count = 0
        self._last_new_reveals = 0
        self.total_new_reveals = 0.0

        return {
            "obs": self._build_obs(),
            "action_mask": self._compute_action_mask(),
            "aux": self._build_aux(),
        }

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        action = int(action)
        cell = action % (self.H * self.W)
        r, c = divmod(cell, self.W)

        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        outcome: Optional[str] = None
        self._last_new_reveals = 0

        total_cells = self.H * self.W
        total_safe = total_cells - int(self.cfg.mine_count)

        if not self.revealed[r, c]:
            if not self.first_click_done:
                self._place_mines_safe(first_click_rc=(r, c))
                self.first_click_done = True

            if self.mine_mask[r, c]:
                self.revealed[r, c] = True
                done = True
                outcome = "loss"
                reward += float(self.cfg.loss_reward)
            else:
                newly_revealed = self._reveal_with_flood_fill(r, c)
                deductions_revealed, _ = self._apply_deductions()
                total_new = newly_revealed + deductions_revealed
                self._last_new_reveals = total_new
                self.total_new_reveals += float(total_new)
                reward += float(total_new) / float(total_cells)
                if int(self.revealed.sum()) >= total_safe:
                    done = True
                    outcome = "win"
                    reward += float(self.cfg.win_reward)
        else:
            # Invalid action (already revealed) -> no reward change, episode continues
            pass

        self.step_count += 1

        info["outcome"] = outcome

        obs_dict = {
            "obs": self._build_obs(),
            "action_mask": self._compute_action_mask(),
            "aux": self._build_aux(),
        }
        return obs_dict, float(reward), bool(done), info

    @property
    def action_space(self) -> int:
        return self.A

    @property
    def obs_channels(self) -> int:
        # revealed, flags, one-hot 0..8
        return 11

    # ------------------------ Internal helpers ------------------------
    def _build_aux(self) -> Dict[str, Any]:
        total_cells = self.H * self.W
        revealed_frac = float(int(self.revealed.sum()) / max(1, total_cells))
        return {
            "step": int(self.step_count),
            "last_new_reveals": int(self._last_new_reveals),
            "revealed_frac": revealed_frac,
        }

    def _build_obs(self) -> np.ndarray:
        # channels: revealed, flags, one-hot numbers
        revealed = self.revealed.astype(np.float32)
        flags = self.flags.astype(np.float32)

        onehot = np.zeros((9, self.H, self.W), dtype=np.float32)
        if self.first_click_done:
            # activate one-hot only where revealed
            counts = self.adjacent_counts
            for k in range(9):
                mask_k = (counts == k) & self.revealed
                if mask_k.any():
                    onehot[k][mask_k] = 1.0
        # else keep zeros (no numbers known yet)

        obs = np.concatenate([
            revealed[None, ...],
            flags[None, ...],
            onehot,
        ], axis=0)
        return obs.astype(np.float32, copy=False)

    def _compute_action_mask(self) -> np.ndarray:
        # Only unrevealed cells are valid actions
        unrevealed = ~self.revealed
        return unrevealed.reshape(-1).astype(bool, copy=False)

    def _reveal_with_flood_fill(self, r: int, c: int) -> int:
        """Reveal cell (r,c). If zero, BFS flood-fill reveals connected zero region
        and its border number cells. Respects flags (does not auto-unflag).
        Returns number of newly revealed safe cells.
        """
        if self.revealed[r, c] or self.flags[r, c]:
            return 0

        newly_revealed = 0
        q: deque[Tuple[int, int]] = deque()
        q.append((r, c))

        while q:
            rr, cc = q.popleft()
            if self.revealed[rr, cc] or self.flags[rr, cc]:
                continue
            if self.mine_mask[rr, cc]:
                # safety: should not happen for non-mine reveals unless inconsistent call
                continue
            self.revealed[rr, cc] = True
            newly_revealed += 1

            if self.adjacent_counts[rr, cc] == 0:
                for nr, nc in self._neighbors(rr, cc):
                    if not self.revealed[nr, nc] and not self.flags[nr, nc]:
                        if not self.mine_mask[nr, nc]:
                            q.append((nr, nc))
        return newly_revealed

    def _maybe_auto_chord(self) -> Tuple[int, int]:
        """If a revealed number cell has adjacent flags equal to its number,
        auto-reveal its other unrevealed, unflagged neighbors. Returns count of newly revealed.
        """
        total_new = 0
        events = 0
        number_cells = np.argwhere(self.revealed & (self.adjacent_counts > 0))
        for rr, cc in number_cells:
            num = int(self.adjacent_counts[rr, cc])
            neigh = self._neighbors(rr, cc)
            flag_cnt = 0
            to_reveal = []
            for nr, nc in neigh:
                if self.flags[nr, nc] and not self.revealed[nr, nc]:
                    flag_cnt += 1
                if (not self.flags[nr, nc]) and (not self.revealed[nr, nc]) and (not self.mine_mask[nr, nc]):
                    to_reveal.append((nr, nc))
            if flag_cnt == num and len(to_reveal) > 0:
                events += 1
                for tr, tc in to_reveal:
                    total_new += self._reveal_with_flood_fill(tr, tc)
        return total_new, events

    def _apply_deductions(self) -> Tuple[int, int]:
        if not self.first_click_done:
            return 0, 0
        total_revealed = 0
        total_flagged = 0
        while True:
            progress = False
            moves = forced_moves(self)
            for act_type, idx in moves:
                r, c = divmod(idx, self.W)
                if act_type == "flag":
                    if not self.flags[r, c]:
                        self.flags[r, c] = True
                        total_flagged += 1
                        progress = True
                else:  # reveal
                    if not self.revealed[r, c] and not self.flags[r, c]:
                        if self.mine_mask[r, c]:
                            continue
                        total_revealed += self._reveal_with_flood_fill(r, c)
                        progress = True
            chord_new, chord_events = self._maybe_auto_chord()
            if chord_new > 0:
                total_revealed += chord_new
                progress = True
            if not progress:
                break
        return total_revealed, total_flagged

    def _place_mines_safe(self, first_click_rc: Tuple[int, int]) -> None:
        r0, c0 = first_click_rc
        H, W = self.H, self.W
        total = H * W
        mines = int(self.cfg.mine_count)

        forbidden = np.zeros((H, W), dtype=bool)
        forbidden[r0, c0] = True
        if self.cfg.guarantee_safe_neighborhood:
            for rr, cc in self._neighbors(r0, c0, include_self=False, include_diagonals=True):
                forbidden[rr, cc] = True

        allowed_indices = (~forbidden).reshape(-1).nonzero()[0]
        if len(allowed_indices) < mines:
            # In tiny boards with strong safety, relax to at least exclude the clicked cell
            forbidden = np.zeros((H, W), dtype=bool)
            forbidden[r0, c0] = True
            allowed_indices = (~forbidden).reshape(-1).nonzero()[0]

        mine_positions = self.rng.choice(allowed_indices, size=mines, replace=False)
        self.mine_mask = np.zeros((H, W), dtype=bool)
        self.mine_mask.reshape(-1)[mine_positions] = True
        self.adjacent_counts = self._compute_adjacent_counts(self.mine_mask)

    def _compute_adjacent_counts(self, mine_mask: np.ndarray) -> np.ndarray:
        counts = np.zeros_like(mine_mask, dtype=np.uint8)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                shifted = self._shift_bool(mine_mask, dr, dc)
                counts = counts + shifted.astype(np.uint8)
        return counts

    def _neighbors(
        self,
        r: int,
        c: int,
        include_self: bool = False,
        include_diagonals: bool = True,
    ) -> Tuple[Tuple[int, int], ...]:
        neigh = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if not include_self and dr == 0 and dc == 0:
                    continue
                if not include_diagonals and (abs(dr) + abs(dc) != 1):
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.H and 0 <= cc < self.W:
                    neigh.append((rr, cc))
        return tuple(neigh)

    def _shift_bool(self, arr: np.ndarray, dr: int, dc: int) -> np.ndarray:
        H, W = arr.shape
        out = np.zeros_like(arr, dtype=bool)
        r_src_start = max(0, -dr)
        r_src_end = min(H, H - dr)  # exclusive
        c_src_start = max(0, -dc)
        c_src_end = min(W, W - dc)

        r_dst_start = max(0, dr)
        r_dst_end = r_dst_start + (r_src_end - r_src_start)
        c_dst_start = max(0, dc)
        c_dst_end = c_dst_start + (c_src_end - c_src_start)

        if r_src_start < r_src_end and c_src_start < c_src_end:
            out[r_dst_start:r_dst_end, c_dst_start:c_dst_end] = arr[
                r_src_start:r_src_end, c_src_start:c_src_end
            ]
        return out


class VecMinesweeper:
    """Batched wrapper that keeps N independent envs on CPU."""

    def __init__(self, num_envs: int, cfg: EnvConfig, seed: int = 0):
        assert num_envs > 0
        self.cfg = cfg
        self.num_envs = int(num_envs)
        base_rng = np.random.default_rng(seed)
        seeds = base_rng.integers(0, 2**31 - 1, size=self.num_envs, dtype=np.int64)
        self.envs = [MinesweeperEnv(cfg, int(seeds[i])) for i in range(self.num_envs)]

    def reset(self) -> Dict[str, np.ndarray]:
        obs_list = []
        mask_list = []
        for e in self.envs:
            d = e.reset()
            obs_list.append(d["obs"])  # [C,H,W]
            mask_list.append(d["action_mask"])  # [A]
        obs = np.stack(obs_list, axis=0)
        mask = np.stack(mask_list, axis=0)
        return {"obs": obs, "action_mask": mask}

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, Any]]:
        assert actions.shape == (self.num_envs,)
        next_obs = []
        next_mask = []
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        dones = np.zeros((self.num_envs,), dtype=bool)
        infos: Dict[str, Any] = {
            "aux": [],
            "outcome": [],
            "done": [],
        }

        for i, e in enumerate(self.envs):
            d_step, r, done, info_step = e.step(int(actions[i]))
            outcome = info_step.get("outcome") if done else None
            aux_step = d_step.get("aux", {})
            # auto-reset done envs to streamline rollouts
            d = d_step
            if done:
                d = e.reset()
            next_obs.append(d["obs"])  # [C,H,W]
            next_mask.append(d["action_mask"])  # [A]
            rewards[i] = r
            dones[i] = done
            infos["aux"].append(aux_step)
            infos["outcome"].append(outcome)
            infos["done"].append(bool(done))

        batch = {
            "obs": np.stack(next_obs, axis=0),
            "action_mask": np.stack(next_mask, axis=0),
        }
        return batch, rewards, dones, infos

    def action_space(self) -> int:
        return self.envs[0].action_space

    def obs_channels(self) -> int:
        return self.envs[0].obs_channels
