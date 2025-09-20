from __future__ import annotations

import numpy as np

try:  # Optional acceleration if numba is available
    from numba import njit  # type: ignore

    HAS_ENV_NUMBA = True
except Exception:  # pragma: no cover - numba not installed
    njit = None  # type: ignore
    HAS_ENV_NUMBA = False


if HAS_ENV_NUMBA:

    @njit(cache=True)  # type: ignore[misc]
    def flood_fill_reveal(
        revealed: np.ndarray,
        flags: np.ndarray,
        mine_mask: np.ndarray,
        adjacent_counts: np.ndarray,
        start_r: int,
        start_c: int,
    ) -> int:
        H, W = revealed.shape
        if revealed[start_r, start_c] or flags[start_r, start_c]:
            return 0
        if mine_mask[start_r, start_c]:
            return 0

        max_cells = H * W
        queue_r = np.empty(max_cells, dtype=np.int64)
        queue_c = np.empty(max_cells, dtype=np.int64)
        queued = np.zeros((H, W), dtype=np.bool_)

        head = 0
        tail = 0
        queue_r[tail] = start_r
        queue_c[tail] = start_c
        tail += 1
        queued[start_r, start_c] = True

        newly_revealed = 0

        while head < tail:
            rr = int(queue_r[head])
            cc = int(queue_c[head])
            head += 1

            if revealed[rr, cc] or flags[rr, cc]:
                continue
            if mine_mask[rr, cc]:
                continue

            revealed[rr, cc] = True
            newly_revealed += 1

            if adjacent_counts[rr, cc] == 0:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if nr < 0 or nr >= H or nc < 0 or nc >= W:
                            continue
                        if queued[nr, nc]:
                            continue
                        if revealed[nr, nc] or flags[nr, nc] or mine_mask[nr, nc]:
                            continue

                        queue_r[tail] = nr
                        queue_c[tail] = nc
                        tail += 1
                        queued[nr, nc] = True

        return newly_revealed

else:  # pragma: no cover - fallback used only when numba missing

    def flood_fill_reveal(*args, **kwargs) -> int:  # type: ignore[no-untyped-def]
        raise RuntimeError("Numba flood fill requested but numba is not available")
