#!/usr/bin/env python3
"""Heatmap of the 113x113 modular addition table showing diagonal class structure.

Visualises (a + b) mod 113 for all a, b in [0, 112].  Same-output classes
lie along wrapping diagonals — this plot makes that structure explicit.
"""

import pathlib, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

P = 113
OUT_DIR = (pathlib.Path(__file__).resolve().parent.parent /
           "outputs" / "advisor_core_figures")


def main():
    # Build the addition table
    a = np.arange(P)
    table = (a[:, None] + a[None, :]) % P          # shape (113, 113)

    # --- Full heatmap ---
    # Use a cyclic colourmap so class 0 and class 112 are close in hue,
    # emphasising the wrapping-diagonal structure.
    fig, ax = plt.subplots(figsize=(7, 6.5))
    cmap = plt.cm.twilight_shifted
    im = ax.imshow(table, cmap=cmap, origin="upper", interpolation="nearest")

    # Tick marks at multiples of 28 (roughly quarters)
    ticks = list(range(0, P, 28)) + [P - 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("$b$", fontsize=13)
    ax.set_ylabel("$a$", fontsize=13)
    ax.set_title(
        "$(a + b)$ mod $113$  — same output classes lie on diagonals",
        fontsize=12, pad=10)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("$(a+b)$ mod $113$", fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "modular_addition_table_diagonals.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Highlighted-diagonal inset (single class) ---
    # Show a binary mask for one output class to make the diagonal explicit.
    cls_highlight = 17                              # arbitrary representative class
    mask = (table == cls_highlight).astype(float)

    fig2, ax2 = plt.subplots(figsize=(7, 6.5))
    # Background: faint full table
    ax2.imshow(table, cmap="gray", alpha=0.15, origin="upper",
               interpolation="nearest")
    # Overlay: highlighted diagonal
    masked = np.ma.masked_where(mask == 0, mask)
    ax2.imshow(masked, cmap=ListedColormap(["#C44E52"]), origin="upper",
               interpolation="nearest", vmin=0, vmax=1)

    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    ax2.set_xlabel("$b$", fontsize=13)
    ax2.set_ylabel("$a$", fontsize=13)
    ax2.set_title(
        f"Output class {cls_highlight}: all $(a,b)$ with "
        f"$(a+b) \\equiv {cls_highlight}$  (mod $113$)",
        fontsize=12, pad=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig2.tight_layout()
    path2 = OUT_DIR / "modular_addition_single_diagonal.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {path2}")


if __name__ == "__main__":
    main()
