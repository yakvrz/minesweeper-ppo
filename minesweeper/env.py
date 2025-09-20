from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple, Optional

import numpy as np
from collections import deque

from .env_numba import HAS_ENV_NUMBA, flood_fill_reveal
from .rules import forced_moves


@dataclass
class SolverPreset(str, Enum):
    ZF = "zf"
    ZF_CHORD = "zf_chord"
    ZF_CHORD_ALL_SAFE = "zf_chord_all_safe"
    ZF_CHORD_ALL_SAFE_ALL_MINE = "zf_chord_all_safe_all_mine"
    ZF_CHORD_ALL_SAFE_ALL_MINE_PAIRWISE = "zf_chord_all_safe_all_mine_pairwise"


@dataclass
class EnvConfig:
    H: int = 8
    W: int = 8
    mine_count: int = 10
    guarantee_safe_neighborhood: bool = True
    use_pair_constraints: bool | None = None  # deprecated
    solver_preset: str = SolverPreset.ZF_CHORD_ALL_SAFE_ALL_MINE_PAIRWISE.value

    win_reward: float = 1.0
    loss_reward: float = -1.0
    step_penalty: float = 1e-4
    progress_scale: float = 0.6

    include_flags_channel: bool = False
    include_frontier_channel: bool = False
    include_remaining_mines_channel: bool = False
    include_progress_channel: bool = False


class MinesweeperEnv:
    """Single Minesweeper environment running on CPU with NumPy state.

    Observation encoding (C,H,W) is dynamic:
      - revealed mask {0,1}
      - optional flags mask {0,1} (if include_flags_channel)
      - nine one-hot planes for adjacent counts 0..8 (active only where revealed=1)
      - optional helper channels (frontier, remaining mines ratio, progress scalar)
    """

    def __init__(self, cfg: EnvConfig, seed: int = 0):
        self.cfg = cfg
        self.H = int(cfg.H)
        self.W = int(cfg.W)
        self.A = self.H * self.W

        self.rng = np.random.default_rng(seed)

        # Pre-compute observation bookkeeping
        self._obs_channel_count = self._calc_obs_channels()
        self._obs_buffer = np.zeros((self._obs_channel_count, self.H, self.W), dtype=np.float32)
        self._onehot_buffer = np.zeros((9, self.H, self.W), dtype=np.float32)
        self._frontier_buffer = np.zeros((self.H, self.W), dtype=bool)
        self._pad_buffer = np.zeros((self.H + 2, self.W + 2), dtype=bool)
        self._shift_buffer = np.zeros((self.H, self.W), dtype=bool)
        self._forbidden_buffer = np.zeros((self.H, self.W), dtype=bool)
        self._mine_pad_buffer = np.zeros((self.H + 2, self.W + 2), dtype=np.uint8)
        self._adjacent_work = np.zeros((self.H, self.W), dtype=np.uint8)
        self._neighbor_offsets = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        )

        # Runtime state (re-used between resets)
        self.mine_mask = np.zeros((self.H, self.W), dtype=bool)
        self.adjacent_counts = np.zeros((self.H, self.W), dtype=np.uint8)
        self.revealed = np.zeros((self.H, self.W), dtype=bool)
        self.flags = np.zeros((self.H, self.W), dtype=bool)
        self.first_click_done = False
        self.step_count = 0
        self._last_new_reveals = 0
        self.total_new_reveals = 0.0

        self.reset()

    # ------------------------ Public API ------------------------
    def _calc_obs_channels(self) -> int:
        channels = 1  # revealed
        if self.cfg.include_flags_channel:
            channels += 1
        channels += 9  # counts 0..8
        if self.cfg.include_frontier_channel:
            channels += 1
        if self.cfg.include_remaining_mines_channel:
            channels += 1
        if self.cfg.include_progress_channel:
            channels += 1
        return channels

    def reset(self) -> Dict[str, Any]:
        self.mine_mask.fill(False)
        self.adjacent_counts.fill(0)
        self.revealed.fill(False)
        self.flags.fill(False)
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
                if newly_revealed > 0:
                    deductions_revealed, _ = self._apply_deductions()
                else:
                    deductions_revealed = 0
                total_new = newly_revealed + deductions_revealed
                self._last_new_reveals = total_new
                self.total_new_reveals += float(total_new)
                reward += float(self.cfg.progress_scale) * float(total_new) / float(total_cells)
                if int(self.revealed.sum()) >= total_safe:
                    done = True
                    outcome = "win"
                    reward += float(self.cfg.win_reward)
        else:
            # Invalid action (already revealed) -> no reward change, episode continues
            pass

        reward -= float(self.cfg.step_penalty)
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
        return self._obs_channel_count

    # ------------------------ Internal helpers ------------------------
    def _build_aux(self) -> Dict[str, Any]:
        total_cells = self.H * self.W
        revealed_frac = float(int(self.revealed.sum()) / max(1, total_cells))
        return {
            "step": int(self.step_count),
            "last_new_reveals": int(self._last_new_reveals),
            "revealed_frac": revealed_frac,
            "toggles": 0,
        }

    def _build_obs(self) -> np.ndarray:
        obs = self._obs_buffer
        ch = 0

        np.copyto(obs[ch], self.revealed, casting="unsafe")
        ch += 1

        if self.cfg.include_flags_channel:
            np.copyto(obs[ch], self.flags, casting="unsafe")
            ch += 1

        onehot = self._onehot_buffer
        onehot.fill(0.0)
        if self.first_click_done:
            rr, cc = np.nonzero(self.revealed)
            if rr.size:
                counts = self.adjacent_counts[rr, cc]
                onehot[counts, rr, cc] = 1.0
        obs[ch : ch + 9] = onehot
        ch += 9

        if self.cfg.include_frontier_channel:
            frontier = self._compute_frontier()
            np.copyto(obs[ch], frontier, casting="unsafe")
            ch += 1

        if self.cfg.include_remaining_mines_channel:
            remaining_ratio = self._remaining_mines_ratio()
            obs[ch].fill(float(remaining_ratio))
            ch += 1

        if self.cfg.include_progress_channel:
            safe_total = max(1, self.A - int(self.cfg.mine_count))
            progress = float(self.revealed.sum()) / float(safe_total)
            obs[ch].fill(progress)
            ch += 1

        return obs.copy()

    def _compute_action_mask(self) -> np.ndarray:
        # Only unrevealed cells are valid actions
        unrevealed = ~self.revealed
        return unrevealed.reshape(-1).astype(bool, copy=False)

    def _reveal_with_flood_fill(self, r: int, c: int) -> int:
        """Reveal cell (r,c) with flood-fill expansion for zero tiles."""
        if self.revealed[r, c] or self.flags[r, c]:
            return 0

        if HAS_ENV_NUMBA:
            return int(
                flood_fill_reveal(
                    self.revealed,
                    self.flags,
                    self.mine_mask,
                    self.adjacent_counts,
                    int(r),
                    int(c),
                )
            )

        return self._reveal_with_flood_fill_python(r, c)

    def _reveal_with_flood_fill_python(self, r: int, c: int) -> int:
        """Python fallback flood-fill used when numba acceleration is unavailable."""
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
                continue
            self.revealed[rr, cc] = True
            newly_revealed += 1

            if self.adjacent_counts[rr, cc] == 0:
                for dr, dc in self._neighbor_offsets:
                    nr = rr + dr
                    nc = cc + dc
                    if nr < 0 or nr >= self.H or nc < 0 or nc >= self.W:
                        continue
                    if self.revealed[nr, nc] or self.flags[nr, nc] or self.mine_mask[nr, nc]:
                        continue
                    q.append((nr, nc))
        return newly_revealed

    def _apply_deductions(self) -> Tuple[int, int]:
        if not self.first_click_done:
            return 0, 0

        total_revealed = 0
        total_flagged = 0

        while True:
            moves = forced_moves(self)
            if not moves:
                break

            progress = False
            for action, idx in moves:
                r, c = divmod(int(idx), self.W)
                if action == "flag":
                    if not self.flags[r, c]:
                        self.flags[r, c] = True
                        total_flagged += 1
                        progress = True
                else:  # reveal
                    if not self.revealed[r, c] and not self.mine_mask[r, c]:
                        newly = self._reveal_with_flood_fill(r, c)
                        if newly > 0:
                            total_revealed += newly
                            progress = True

            if not progress:
                break

        return total_revealed, total_flagged

    def _compute_frontier(self) -> np.ndarray:
        frontier = self._frontier_buffer
        frontier.fill(False)
        if not self.first_click_done:
            return frontier
        number_cells = self.revealed & (self.adjacent_counts > 0)
        if not number_cells.any():
            return frontier
        pad = self._pad_buffer
        pad.fill(False)
        pad[1:-1, 1:-1] = number_cells

        np.logical_or(frontier, pad[:-2, :-2], out=frontier)
        np.logical_or(frontier, pad[:-2, 1:-1], out=frontier)
        np.logical_or(frontier, pad[:-2, 2:], out=frontier)
        np.logical_or(frontier, pad[1:-1, :-2], out=frontier)
        np.logical_or(frontier, pad[1:-1, 2:], out=frontier)
        np.logical_or(frontier, pad[2:, :-2], out=frontier)
        np.logical_or(frontier, pad[2:, 1:-1], out=frontier)
        np.logical_or(frontier, pad[2:, 2:], out=frontier)
        unknown = (~self.revealed) & (~self.flags)
        frontier &= unknown
        return frontier

    def _remaining_mines_ratio(self) -> float:
        total_mines = int(self.cfg.mine_count)
        if total_mines <= 0:
            return 0.0
        flagged_true = int((self.flags & self.mine_mask).sum()) if self.first_click_done else 0
        remaining = max(0, total_mines - flagged_true)
        return float(remaining) / float(total_mines)

    def _place_mines_safe(self, first_click_rc: Tuple[int, int]) -> None:
        r0, c0 = first_click_rc
        H, W = self.H, self.W
        total = H * W
        mines = int(self.cfg.mine_count)

        forbidden = self._forbidden_buffer
        forbidden.fill(False)
        if self.cfg.guarantee_safe_neighborhood:
            pad = self._pad_buffer
            pad.fill(False)
            pad[1 + r0, 1 + c0] = True
            np.logical_or(forbidden, pad[:-2, :-2], out=forbidden)
            np.logical_or(forbidden, pad[:-2, 1:-1], out=forbidden)
            np.logical_or(forbidden, pad[:-2, 2:], out=forbidden)
            np.logical_or(forbidden, pad[1:-1, :-2], out=forbidden)
            np.logical_or(forbidden, pad[1:-1, 2:], out=forbidden)
            np.logical_or(forbidden, pad[2:, :-2], out=forbidden)
            np.logical_or(forbidden, pad[2:, 1:-1], out=forbidden)
            np.logical_or(forbidden, pad[2:, 2:], out=forbidden)
        forbidden[r0, c0] = True

        allowed_indices = np.flatnonzero(~forbidden)
        if len(allowed_indices) < mines:
            # In tiny boards with strong safety, relax to at least exclude the clicked cell
            forbidden.fill(False)
            forbidden[r0, c0] = True
            allowed_indices = np.flatnonzero(~forbidden)

        mine_positions = self.rng.choice(allowed_indices, size=mines, replace=False)
        self.mine_mask.fill(False)
        self.mine_mask.reshape(-1)[mine_positions] = True
        self._compute_adjacent_counts(self.mine_mask, out=self.adjacent_counts)

    def _compute_adjacent_counts(
        self, mine_mask: np.ndarray, *, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        pad = self._mine_pad_buffer
        pad.fill(0)
        pad[1:-1, 1:-1] = mine_mask

        if out is None:
            counts = np.zeros_like(mine_mask, dtype=np.uint8)
        else:
            counts = out
            counts.fill(0)

        np.add(counts, pad[:-2, :-2], out=counts, casting="unsafe")
        np.add(counts, pad[:-2, 1:-1], out=counts, casting="unsafe")
        np.add(counts, pad[:-2, 2:], out=counts, casting="unsafe")
        np.add(counts, pad[1:-1, :-2], out=counts, casting="unsafe")
        np.add(counts, pad[1:-1, 2:], out=counts, casting="unsafe")
        np.add(counts, pad[2:, :-2], out=counts, casting="unsafe")
        np.add(counts, pad[2:, 1:-1], out=counts, casting="unsafe")
        np.add(counts, pad[2:, 2:], out=counts, casting="unsafe")
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

    def _shift_bool(self, arr: np.ndarray, dr: int, dc: int, out: Optional[np.ndarray] = None) -> np.ndarray:
        H, W = arr.shape
        if out is None:
            out = np.zeros_like(arr, dtype=bool)
        else:
            out.fill(False)
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
