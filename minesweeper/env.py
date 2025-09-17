from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy as np
from collections import deque


@dataclass
class EnvConfig:
    H: int = 8
    W: int = 8
    mine_count: int = 10
    guarantee_safe_neighborhood: bool = True

    progress_reward: float = 0.01
    win_reward: float = 1.0
    loss_reward: float = -1.0
    step_penalty: float = 1e-4
    invalid_penalty: float = 1e-3

    flag_correct_reward: float = 0.002
    flag_incorrect_reward: float = -0.002
    use_flag_shaping: bool = False


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
        self.A = 2 * self.H * self.W

        self.rng = np.random.default_rng(seed)

        # Runtime state
        self.mine_mask: np.ndarray
        self.adjacent_counts: np.ndarray
        self.revealed: np.ndarray
        self.flags: np.ndarray
        self.first_click_done: bool
        self.step_count: int
        self._last_new_reveals: int = 0

        # Optional curriculum toggle: restrict reveal actions to frontier
        self.frontier_only_reveal: bool = False

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

        return {
            "obs": self._build_obs(),
            "action_mask": self._compute_action_mask(),
            "aux": self._build_aux(),
        }

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        action = int(action)
        cell = action % (self.H * self.W)
        r, c = divmod(cell, self.W)
        is_flag = action >= (self.H * self.W)

        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        # Invalid: cannot act on revealed cells
        if self.revealed[r, c]:
            reward += -float(self.cfg.invalid_penalty)
            # no state change
        else:
            if is_flag:
                # toggle flag on unrevealed cell
                before = bool(self.flags[r, c])
                self.flags[r, c] = ~before
                # optional shaping only after mines exist
                if self.cfg.use_flag_shaping and self.first_click_done:
                    now_flagged = bool(self.flags[r, c])
                    if now_flagged:
                        # added a flag
                        if self.mine_mask[r, c]:
                            reward += float(self.cfg.flag_correct_reward)
                        else:
                            reward += -float(self.cfg.flag_incorrect_reward)
                    else:
                        # removed a flag: no reward
                        pass
            else:
                # reveal
                if not self.first_click_done:
                    self._place_mines_safe(first_click_rc=(r, c))
                    self.first_click_done = True

                if self.mine_mask[r, c]:
                    # terminal loss
                    self.revealed[r, c] = True
                    reward += float(self.cfg.loss_reward)
                    done = True
                else:
                    newly_revealed = self._reveal_with_flood_fill(r, c)
                    # shaping by number of newly revealed safe cells
                    reward += float(self.cfg.progress_reward) * float(newly_revealed)
                    self._last_new_reveals = newly_revealed

                    # win condition: all safe cells revealed
                    total_safe = self.H * self.W - int(self.cfg.mine_count)
                    if int(self.revealed.sum()) >= total_safe:
                        reward += float(self.cfg.win_reward)
                        done = True

        # step penalty always applied
        reward += -float(self.cfg.step_penalty)
        self.step_count += 1

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
        if not self.first_click_done:
            remaining_mines_ratio = 1.0
        else:
            # Estimate remaining mines as total_mines - flagged_true (not perfect but informative)
            total_mines = int(self.cfg.mine_count)
            flagged = int((self.flags & self.mine_mask).sum())
            remaining_mines = max(0, total_mines - flagged)
            remaining_mines_ratio = remaining_mines / max(1, total_mines)
        return {
            "step": int(self.step_count),
            "last_new_reveals": int(self._last_new_reveals),
            "remaining_mines_ratio": float(remaining_mines_ratio),
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
        # Reveal valid on unrevealed cells (optionally frontier-only)
        unrevealed = ~self.revealed

        if self.frontier_only_reveal and self.first_click_done:
            frontier = self._compute_frontier()
            reveal_valid = unrevealed & frontier
            # If no frontier exists (early or trivial board), fall back to any unrevealed
            if not reveal_valid.any():
                reveal_valid = unrevealed
        else:
            reveal_valid = unrevealed

        flag_valid = unrevealed

        reveal_mask = reveal_valid.reshape(-1)
        flag_mask = flag_valid.reshape(-1)
        full_mask = np.concatenate([reveal_mask, flag_mask], axis=0)
        return full_mask.astype(bool, copy=False)

    def _compute_frontier(self) -> np.ndarray:
        # Unknown cells adjacent to at least one revealed number
        if not self.first_click_done:
            return np.zeros((self.H, self.W), dtype=bool)
        # cells with revealed numbers
        number_cells = self.revealed & (self.adjacent_counts > 0)
        if not number_cells.any():
            return np.zeros((self.H, self.W), dtype=bool)
        frontier = np.zeros((self.H, self.W), dtype=bool)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                shifted = self._shift_bool(number_cells, dr, dc)
                frontier |= shifted
        # unknown = not revealed and not flagged
        unknown = (~self.revealed) & (~self.flags)
        frontier &= unknown
        return frontier

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
    """Batched wrapper that keeps N independent envs on CPU.

    The API mirrors a standard vectorized env with NumPy arrays.
    """

    def __init__(self, num_envs: int, cfg: EnvConfig, seed: int = 0):
        assert num_envs > 0
        self.cfg = cfg
        self.num_envs = int(num_envs)
        # Derive different seeds to decorrelate
        base_rng = np.random.default_rng(seed)
        seeds = base_rng.integers(0, 2**31 - 1, size=self.num_envs, dtype=np.int64)
        self.envs = [MinesweeperEnv(cfg, int(seeds[i])) for i in range(self.num_envs)]

    def set_frontier_only_reveal(self, enabled: bool) -> None:
        for e in self.envs:
            e.frontier_only_reveal = bool(enabled)

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
        infos: Dict[str, Any] = {"aux": []}

        for i, e in enumerate(self.envs):
            d, r, done, info = e.step(int(actions[i]))
            if done:
                # auto-reset done envs to streamline rollouts
                d = e.reset()
            next_obs.append(d["obs"])  # [C,H,W]
            next_mask.append(d["action_mask"])  # [A]
            rewards[i] = r
            dones[i] = done
            infos["aux"].append(d.get("aux", {}))

        batch = {
            "obs": np.stack(next_obs, axis=0),
            "action_mask": np.stack(next_mask, axis=0),
        }
        return batch, rewards, dones, infos

    def action_space(self) -> int:
        return self.envs[0].action_space

    def obs_channels(self) -> int:
        return self.envs[0].obs_channels


