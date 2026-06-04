"""
analysis/erank_entropy_illustration.py

Pedagogical figures for the thesis: two normalised singular-value
distributions that bracket the eRank scale.

  - Low-eRank example:  mass concentrated on a few singular values
                        => low entropy H(p), low eRank = exp(H)
  - High-eRank example: mass spread across all singular values
                        => high entropy H(p), high eRank = exp(H)

Uses the same definition as analysis/erank.py:
  p_i    = sigma_i / sum_j sigma_j
  H(p)   = -sum_i p_i log p_i           (natural log)
  eRank  = exp(H(p))
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "advisor_core_figures"


def entropy_and_erank(sigma: np.ndarray) -> tuple[float, float]:
    sigma = sigma[sigma > 0]
    p = sigma / sigma.sum()
    H = float(-np.sum(p * np.log(p)))
    return H, float(np.exp(H))


def low_erank_spectrum(d: int = 64) -> np.ndarray:
    # Fast exponential decay: a handful of singular values dominate.
    idx = np.arange(d)
    return np.exp(-idx / 3.0)


def high_erank_spectrum(d: int = 64) -> np.ndarray:
    # Very slow decay, nearly flat across all components.
    idx = np.arange(d)
    return 1.0 - 0.15 * (idx / (d - 1))


def plot_distribution(
    sigma: np.ndarray,
    title: str,
    color: str,
    out_path: Path,
) -> tuple[float, float]:
    p = sigma / sigma.sum()
    H, er = entropy_and_erank(sigma)
    d = len(sigma)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(np.arange(d), p, color=color, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("singular value index $i$")
    ax.set_ylabel(r"normalised $\bar\sigma_i = \sigma_i / \sum_j \sigma_j$")
    ax.set_title(title)
    ax.set_xlim(-0.5, d - 0.5)
    ax.set_ylim(0, max(p) * 1.15)

    txt = (
        f"$H(\\bar\\sigma) = {H:.3f}$\n"
        f"$\\mathrm{{eRank}} = e^{{H}} = {er:.2f}$\n"
        f"$d = {d}$"
    )
    ax.text(
        0.98, 0.95, txt,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6"),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return H, er


def plot_side_by_side(
    sigma_lo: np.ndarray,
    sigma_hi: np.ndarray,
    out_path: Path,
) -> None:
    p_lo = sigma_lo / sigma_lo.sum()
    p_hi = sigma_hi / sigma_hi.sum()
    H_lo, er_lo = entropy_and_erank(sigma_lo)
    H_hi, er_hi = entropy_and_erank(sigma_hi)
    d = len(sigma_lo)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)

    for ax, p, H, er, title, color in [
        (axes[0], p_lo, H_lo, er_lo, "Low eRank — concentrated spectrum", "#c0392b"),
        (axes[1], p_hi, H_hi, er_hi, "High eRank — near-uniform spectrum", "#2c7fb8"),
    ]:
        ax.bar(np.arange(d), p, color=color, edgecolor="black", linewidth=0.3)
        ax.set_xlabel("singular value index $i$")
        ax.set_title(title)
        ax.set_xlim(-0.5, d - 0.5)
        ax.set_ylim(0, max(p_lo.max(), p_hi.max()) * 1.15)
        txt = (
            f"$H(\\bar\\sigma) = {H:.3f}$\n"
            f"$\\mathrm{{eRank}} = {er:.2f}$"
        )
        ax.text(
            0.98, 0.95, txt,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6"),
        )

    axes[0].set_ylabel(r"normalised $\bar\sigma_i$")
    fig.suptitle(
        r"Normalised singular-value distributions: $\mathrm{eRank} = \exp(H(\bar\sigma))$",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = 64
    sigma_lo = low_erank_spectrum(d)
    sigma_hi = high_erank_spectrum(d)

    H_lo, er_lo = plot_distribution(
        sigma_lo,
        title="Low eRank — concentrated spectrum",
        color="#c0392b",
        out_path=OUT_DIR / "erank_illustration_low.png",
    )
    H_hi, er_hi = plot_distribution(
        sigma_hi,
        title="High eRank — near-uniform spectrum",
        color="#2c7fb8",
        out_path=OUT_DIR / "erank_illustration_high.png",
    )
    plot_side_by_side(
        sigma_lo,
        sigma_hi,
        out_path=OUT_DIR / "erank_illustration_side_by_side.png",
    )

    print(f"d = {d}")
    print(f"  low  : H = {H_lo:.4f}, eRank = {er_lo:.3f}")
    print(f"  high : H = {H_hi:.4f}, eRank = {er_hi:.3f}  (max possible = {d})")
    print(f"saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
