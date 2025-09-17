from __future__ import annotations

from typing import List, Tuple
import numpy as np


def forced_moves(state) -> List[Tuple[str, int]]:
    """Compute forced moves from a Minesweeper state using classic constraints.

    The state is expected to expose arrays:
      - revealed: (H,W) bool
      - flags: (H,W) bool
      - adjacent_counts: (H,W) uint8

    Returns a list of ("reveal"|"flag", flat_idx) pairs. Indices are into row-major
    flattened board of size H*W.
    """
    revealed: np.ndarray = state.revealed
    flags: np.ndarray = state.flags
    counts: np.ndarray = state.adjacent_counts

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

    # Deduplicate moves in case multiple constraints imply same action
    if moves:
        # Prefer reveal over flag if contradictory (shouldn't happen for sound rules)
        best = {}
        for act, idx in moves:
            if idx not in best or act == "reveal":
                best[idx] = act
        moves = [(act, idx) for idx, act in best.items()]

    return moves


def _neighbors(H: int, W: int, r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                yield rr, cc


