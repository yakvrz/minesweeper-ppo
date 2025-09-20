from __future__ import annotations

from typing import List, Tuple
import numpy as np

try:  # Optional acceleration if numba is installed.
    from numba import njit  # type: ignore

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - fallback when numba missing.
    njit = None
    _HAS_NUMBA = False

try:  # PyTorch fallback when numba is unavailable.
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch optional for inference envs.
    torch = None
    F = None
_HAS_TORCH = False


_NEIGHBOR_OFFSETS: Tuple[Tuple[int, int], ...] = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

_SOLVER_PRESET_LEVELS = {
    "zf": 1,
    "zf_chord": 2,
    "zf_chord_all_safe": 3,
    "zf_chord_all_safe_all_mine": 4,
    "zf_chord_all_safe_all_mine_pairwise": 5,
    "pairwise": 5,
    "full": 5,
    "default": 5,
}


def _forced_moves_numpy(
    revealed: np.ndarray, flags: np.ndarray, counts: np.ndarray
) -> List[Tuple[str, int]]:
    H, W = revealed.shape
    number_cells = revealed & (counts > 0)
    if not number_cells.any():
        return []

    unknown = (~revealed) & (~flags)
    if not unknown.any():
        return []

    flagged_adj = np.zeros_like(counts, dtype=np.uint8)
    unknown_adj = np.zeros_like(counts, dtype=np.uint8)
    flags_pad = np.pad(flags, 1, constant_values=False)
    unknown_pad = np.pad(unknown, 1, constant_values=False)
    for dr, dc in _NEIGHBOR_OFFSETS:
        flagged_adj += flags_pad[1 + dr : 1 + dr + H, 1 + dc : 1 + dc + W]
        unknown_adj += unknown_pad[1 + dr : 1 + dr + H, 1 + dc : 1 + dc + W]

    rule1 = number_cells & (flagged_adj == counts)
    rule2 = number_cells & ((flagged_adj + unknown_adj == counts) & (unknown_adj > 0))

    reveal_targets = np.zeros_like(revealed, dtype=bool)
    flag_targets = np.zeros_like(revealed, dtype=bool)

    if rule1.any():
        rule1_pad = np.pad(rule1, 1, constant_values=False)
        for dr, dc in _NEIGHBOR_OFFSETS:
            reveal_targets |= rule1_pad[1 + dr : 1 + dr + H, 1 + dc : 1 + dc + W]
        reveal_targets &= unknown

    if rule2.any():
        rule2_pad = np.pad(rule2, 1, constant_values=False)
        for dr, dc in _NEIGHBOR_OFFSETS:
            flag_targets |= rule2_pad[1 + dr : 1 + dr + H, 1 + dc : 1 + dc + W]
        flag_targets &= unknown

    flag_targets &= ~reveal_targets  # give reveal priority

    moves: List[Tuple[str, int]] = []
    if reveal_targets.any():
        rr, cc = np.nonzero(reveal_targets)
        for r, c in zip(rr.tolist(), cc.tolist()):
            moves.append(("reveal", int(r * W + c)))
    if flag_targets.any():
        rr, cc = np.nonzero(flag_targets)
        for r, c in zip(rr.tolist(), cc.tolist()):
            moves.append(("flag", int(r * W + c)))

    return moves


def _solver_level_from_state(state) -> int:
    cfg = getattr(state, "cfg", None)
    if cfg is None:
        return 5
    preset = getattr(cfg, "solver_preset", None)
    if preset is not None:
        key = str(preset).strip().lower()
        if key.isdigit():
            try:
                level = int(key)
                return max(1, min(5, level))
            except Exception:
                pass
        level = _SOLVER_PRESET_LEVELS.get(key)
        if level is not None:
            return level
    use_pair = getattr(cfg, "use_pair_constraints", None)
    if use_pair is not None:
        return 5 if bool(use_pair) else 4
    return 5


def _has_flag_neighbor(flags: np.ndarray, idx: int, width: int) -> bool:
    H, W = flags.shape
    r, c = divmod(idx, width)
    for dr, dc in _NEIGHBOR_OFFSETS:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < H and 0 <= cc < W and flags[rr, cc]:
            return True
    return False


def _dedupe_moves(moves: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    best = {}
    for action, idx in moves:
        if idx not in best or action == "reveal":
            best[idx] = action
    return [(action, idx) for idx, action in best.items()]


def _split_moves(state, moves: List[Tuple[str, int]]) -> tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    safe_all: List[Tuple[str, int]] = []
    safe_chord: List[Tuple[str, int]] = []
    mine_moves: List[Tuple[str, int]] = []
    flags = state.flags
    width = state.W
    for action, idx in moves:
        idx_int = int(idx)
        if action == "reveal":
            safe_all.append((action, idx_int))
            if _has_flag_neighbor(flags, idx_int, width):
                safe_chord.append((action, idx_int))
        elif action == "flag":
            mine_moves.append((action, idx_int))
    return safe_all, safe_chord, mine_moves


def forced_moves(state) -> List[Tuple[str, int]]:
    """Compute forced moves from a Minesweeper state using classic constraints.

    The state is expected to expose arrays:
      - revealed: (H,W) bool
      - flags: (H,W) bool
      - adjacent_counts: (H,W) uint8

    Returns a list of ("reveal"|"flag", flat_idx) pairs. Indices are into row-major
    flattened board of size H*W. NumPy/Numba is used for speed when available.
    """

    revealed: np.ndarray = state.revealed
    flags: np.ndarray = state.flags
    counts: np.ndarray = state.adjacent_counts

    if _HAS_NUMBA:
        actions, indices = _forced_moves_numba(
            np.ascontiguousarray(revealed, dtype=np.bool_),
            np.ascontiguousarray(flags, dtype=np.bool_),
            np.ascontiguousarray(counts, dtype=np.uint8),
        )
        return [
            ("reveal" if act == 2 else "flag", int(idx))
            for act, idx in zip(actions, indices)
        ]

    level = _solver_level_from_state(state)
    if level <= 1:
        return []

    moves = _forced_moves_numpy(revealed, flags, counts)
    if not moves and _HAS_TORCH:
        actions, indices = _forced_moves_torch(revealed, flags, counts)
        moves = [("reveal" if act == 2 else "flag", int(idx)) for act, idx in zip(actions, indices)]

    if not moves:
        moves = _forced_moves_py(revealed, flags, counts)

    if level >= 5 and moves:
        moves = _apply_pair_constraints(revealed, flags, counts, moves)
    if not moves:
        return []

    safe_all, safe_chord, mine_moves = _split_moves(state, moves)

    if level == 2:
        selected = safe_chord
    elif level == 3:
        selected = safe_all
    else:  # level 4 or 5
        selected = safe_all + mine_moves

    return _dedupe_moves(selected)


def _forced_moves_py(revealed: np.ndarray, flags: np.ndarray, counts: np.ndarray) -> List[Tuple[str, int]]:
    H, W = revealed.shape
    moves: List[Tuple[str, int]] = []

    # Iterate revealed number cells and apply two rules.
    # Rule 1: If number == adjacent_flags -> all other adjacent unknowns are safe (reveal)
    # Rule 2: If number == adjacent_unknowns -> all those unknowns are mines (flag)
    # Unknown = not revealed and not flagged.
    for r in range(H):
        for c in range(W):
            if not revealed[r, c]:
                continue
            n = int(counts[r, c])
            if n == 0:
                continue

            neigh = _neighbors(H, W, r, c)
            adj_flags = 0
            unknown_coords = []
            for rr, cc in neigh:
                if flags[rr, cc]:
                    adj_flags += 1
                elif not revealed[rr, cc]:
                    unknown_coords.append((rr, cc))

            if adj_flags == n and len(unknown_coords) > 0:
                # all unknowns are safe -> reveal
                for (rr, cc) in unknown_coords:
                    idx = rr * W + cc
                    moves.append(("reveal", idx))

            if len(unknown_coords) > 0 and len(unknown_coords) == n - adj_flags:
                # all unknowns must be mines -> flag
                for (rr, cc) in unknown_coords:
                    idx = rr * W + cc
                    moves.append(("flag", idx))

    if moves:
        best = {}
        for act, idx in moves:
            if idx not in best or act == "reveal":
                best[idx] = act
        moves = [(act, idx) for idx, act in best.items()]

    return moves


if _HAS_NUMBA:

    @njit(cache=True)  # type: ignore[misc]
    def _forced_moves_numba(
        revealed: np.ndarray, flags: np.ndarray, counts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        H, W = revealed.shape
        total = H * W
        action_map = np.zeros(total, dtype=np.int8)

        unknown_rows = np.empty(8, dtype=np.int64)
        unknown_cols = np.empty(8, dtype=np.int64)

        for r in range(H):
            for c in range(W):
                if not revealed[r, c]:
                    continue
                n_val = int(counts[r, c])
                if n_val == 0:
                    continue

                flagged = 0
                unknown_count = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr = r + dr
                        cc = c + dc
                        if rr < 0 or rr >= H or cc < 0 or cc >= W:
                            continue
                        if flags[rr, cc]:
                            flagged += 1
                        elif not revealed[rr, cc]:
                            unknown_rows[unknown_count] = rr
                            unknown_cols[unknown_count] = cc
                            unknown_count += 1

                if unknown_count == 0:
                    continue

                if flagged == n_val:
                    for k in range(unknown_count):
                        idx = int(unknown_rows[k] * W + unknown_cols[k])
                        action_map[idx] = 2  # reveal overrides flags

                if flagged + unknown_count == n_val:
                    for k in range(unknown_count):
                        idx = int(unknown_rows[k] * W + unknown_cols[k])
                        if action_map[idx] != 2:  # respect reveal priority
                            action_map[idx] = 1

        move_count = 0
        for i in range(total):
            if action_map[i] != 0:
                move_count += 1

        if move_count == 0:
            return np.empty(0, dtype=np.int8), np.empty(0, dtype=np.int64)

        actions = np.empty(move_count, dtype=np.int8)
        indices = np.empty(move_count, dtype=np.int64)
        ptr = 0
        for i in range(total):
            act_code = action_map[i]
            if act_code != 0:
                actions[ptr] = act_code
                indices[ptr] = i
                ptr += 1

        return actions, indices

else:

    def _forced_moves_numba(
        revealed: np.ndarray, flags: np.ndarray, counts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover - shim
        raise RuntimeError("Numba not available")


def _forced_moves_torch(
    revealed: np.ndarray, flags: np.ndarray, counts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    assert torch is not None and F is not None

    device = torch.device("cpu")
    kernel = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    revealed_bool = torch.from_numpy(np.ascontiguousarray(revealed, dtype=np.bool_)).to(device)
    flags_bool = torch.from_numpy(np.ascontiguousarray(flags, dtype=np.bool_)).to(device)
    counts_int = torch.from_numpy(np.ascontiguousarray(counts, dtype=np.int16)).to(device)

    flags_neigh = F.conv2d(flags_bool.float().unsqueeze(0).unsqueeze(0), kernel, padding=1).squeeze(0).squeeze(0).to(torch.int16)

    unknown_mask = (~revealed_bool) & (~flags_bool)
    unknown_neigh = F.conv2d(unknown_mask.float().unsqueeze(0).unsqueeze(0), kernel, padding=1).squeeze(0).squeeze(0).to(torch.int16)

    number_mask = revealed_bool & (counts_int > 0)

    safe_sources = number_mask & (counts_int == flags_neigh) & (unknown_neigh > 0)
    mine_sources = number_mask & ((counts_int - flags_neigh) == unknown_neigh) & (unknown_neigh > 0)

    if safe_sources.any():
        safe_targets = F.conv2d(safe_sources.float().unsqueeze(0).unsqueeze(0), kernel, padding=1).squeeze(0).squeeze(0) > 0
    else:
        safe_targets = torch.zeros_like(number_mask, dtype=torch.bool)

    if mine_sources.any():
        mine_targets = F.conv2d(mine_sources.float().unsqueeze(0).unsqueeze(0), kernel, padding=1).squeeze(0).squeeze(0) > 0
    else:
        mine_targets = torch.zeros_like(number_mask, dtype=torch.bool)

    reveal_mask = safe_targets & unknown_mask
    flag_mask = mine_targets & unknown_mask & (~reveal_mask)

    action_map = torch.zeros_like(counts_int, dtype=torch.int8)
    action_map[flag_mask] = 1
    action_map[reveal_mask] = 2

    flat = action_map.view(-1)
    sel = flat != 0
    if torch.any(sel):
        actions = flat[sel].to(torch.int8).cpu().numpy()
        indices = torch.nonzero(sel, as_tuple=False).squeeze(1).to(torch.int64).cpu().numpy()
        return actions, indices

    return np.empty(0, dtype=np.int8), np.empty(0, dtype=np.int64)


def _neighbors(H: int, W: int, r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                yield rr, cc
def _apply_pair_constraints(
    revealed: np.ndarray,
    flags: np.ndarray,
    counts: np.ndarray,
    moves: List[Tuple[str, int]],
) -> List[Tuple[str, int]]:
    """Augment move list using simple two-number overlap constraints."""

    if not moves:
        base_moves: List[Tuple[str, int]] = []
    else:
        base_moves = list(moves)

    H, W = revealed.shape
    number_cells = np.transpose(np.nonzero(revealed & (counts > 0)))

    # Precompute unknown neighbours for each number cell
    unknown_list: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    flagged_adj: Dict[Tuple[int, int], int] = {}
    for r, c in number_cells:
        neigh = list(_neighbors(H, W, r, c))
        unknown = [(rr, cc) for (rr, cc) in neigh if not revealed[rr, cc] and not flags[rr, cc]]
        if not unknown:
            continue
        unknown_list[(r, c)] = unknown
        flagged_adj[(r, c)] = sum(1 for (rr, cc) in neigh if flags[rr, cc])

    # Convert existing moves into dictionaries for quick lookup
    move_map: Dict[int, str] = {idx: act for act, idx in base_moves}

    # Pairwise constraint: For two number cells A and B with overlapping unknowns
    # If difference in remaining mines equals size difference of unknown sets, we can ded ded.
    keys = list(unknown_list.keys())
    for i in range(len(keys)):
        r1, c1 = keys[i]
        unknown1 = unknown_list[(r1, c1)]
        count1 = int(counts[r1, c1]) - flagged_adj[(r1, c1)]
        set1 = set(unknown1)
        for j in range(i + 1, len(keys)):
            r2, c2 = keys[j]
            unknown2 = unknown_list[(r2, c2)]
            count2 = int(counts[r2, c2]) - flagged_adj[(r2, c2)]
            set2 = set(unknown2)

            if not set1 or not set2:
                continue

            inter = set1 & set2
            if not inter:
                continue

            diff1 = set1 - set2
            diff2 = set2 - set1

            if not diff1 and not diff2:
                continue

            # If all mines of cell1 are within the intersection, cells unique to cell1 are safe
            if count1 == len(inter) and diff1:
                for (rr, cc) in diff1:
                    idx = rr * W + cc
                    move_map[idx] = "reveal"

            # If all mines of cell2 are within the intersection, cells unique to cell2 are safe
            if count2 == len(inter) and diff2:
                for (rr, cc) in diff2:
                    idx = rr * W + cc
                    move_map[idx] = "reveal"

            # If intersection mines account for difference, unique cells must be mines
            if len(set1) > len(inter) and len(diff1) > 0:
                mines_remaining1 = count1 - len(inter)
                if mines_remaining1 == len(diff1) and mines_remaining1 > 0:
                    for (rr, cc) in diff1:
                        idx = rr * W + cc
                        move_map[idx] = "flag"

            if len(set2) > len(inter) and len(diff2) > 0:
                mines_remaining2 = count2 - len(inter)
                if mines_remaining2 == len(diff2) and mines_remaining2 > 0:
                    for (rr, cc) in diff2:
                        idx = rr * W + cc
                        move_map[idx] = "flag"

    if not move_map:
        return []

    merged_moves = [(act, idx) for idx, act in move_map.items()]
    merged_moves.sort(key=lambda x: x[1])
    return merged_moves
