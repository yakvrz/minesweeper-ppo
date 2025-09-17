from __future__ import annotations

from typing import Optional

import numpy as np


def ascii_board(revealed: np.ndarray, flags: np.ndarray, counts: np.ndarray, mine_mask: Optional[np.ndarray] = None) -> str:
    H, W = revealed.shape
    lines = []
    header = "   " + " ".join(f"{c:2d}" for c in range(W))
    lines.append(header)
    for r in range(H):
        row = [f"{r:2d}"]
        for c in range(W):
            if flags[r, c] and not revealed[r, c]:
                ch = "F "
            elif not revealed[r, c]:
                ch = "# "
            else:
                n = int(counts[r, c])
                ch = f"{n} "
            if mine_mask is not None and mine_mask[r, c]:
                ch = "* "
            row.append(ch)
        lines.append(" ".join(row))
    return "\n".join(lines)


def ascii_from_env(env) -> str:
    return ascii_board(env.revealed, env.flags, env.adjacent_counts)


def plot_heatmap(array: np.ndarray, title: Optional[str] = None, cmap: str = "viridis") -> None:
    """Plot a 2D heatmap (e.g., mine probability) using matplotlib."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    plt.imshow(array, cmap=cmap, interpolation="nearest")
    plt.colorbar()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


