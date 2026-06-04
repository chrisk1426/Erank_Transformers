#!/usr/bin/env python3
"""Layer 1 MLP eRank evidence for standard modular addition.

Grouped bar chart showing R_mid^(1), R_post^(1), and Δ_MLP^(1) per seed,
plus mean ± std overlay. Demonstrates that the layer-1 MLP consistently
expands (Δ_MLP > 0) across all seeds.
"""

import json, pathlib, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED_DIR = (pathlib.Path(__file__).resolve().parent.parent /
            "outputs" / "thesis_additions")
OUT_DIR  = SEED_DIR.parent / "advisor_core_figures"

SEEDS = [0, 1, 2]

COLOR_MID   = "#4C72B0"   # blue  — after attention
COLOR_POST  = "#55A868"   # green — after MLP
COLOR_DELTA = "#E65100"   # dark orange — Δ_MLP (positive, warm)


def load():
    mid, post, delta = [], [], []
    for s in SEEDS:
        p = (SEED_DIR / f"seed_{s}" / "standard_modular_addition" /
             "erank" / "endpoint_erank.json")
        with open(p) as f:
            d = json.load(f)
        mid.append(d["erank"]["layer_1"]["mid"])
        post.append(d["erank"]["layer_1"]["post"])
        delta.append(d["delta_erank_mlp"]["layer_1"])
    return np.array(mid), np.array(post), np.array(delta)


def main():
    mid, post, delta = load()

    metrics = {
        "$R_{\\mathrm{mid}}^{(1)}$":  (mid,   COLOR_MID),
        "$R_{\\mathrm{post}}^{(1)}$": (post,  COLOR_POST),
        "$\\Delta_{\\mathrm{MLP}}^{(1)}$": (delta, COLOR_DELTA),
    }
    labels = list(metrics.keys())
    n_metrics = len(labels)
    n_seeds   = len(SEEDS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   gridspec_kw={"width_ratios": [3, 1.3]})

    # --- Left panel: per-seed bars ---
    bar_w  = 0.22
    x_seed = np.arange(n_seeds)

    for i, (label, (vals, color)) in enumerate(metrics.items()):
        offset = (i - 1) * bar_w
        bars = ax1.bar(x_seed + offset, vals, bar_w, label=label,
                       color=color, alpha=0.85, edgecolor="black",
                       linewidth=0.5)
        for bar, v in zip(bars, vals):
            y_off = 0.3 if v >= 0 else -0.3
            va = "bottom" if v >= 0 else "top"
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + y_off,
                     f"{v:.2f}", ha="center", va=va, fontsize=8)

    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_xticks(x_seed)
    ax1.set_xticklabels([f"Seed {s}" for s in SEEDS], fontsize=11)
    ax1.set_ylabel("eRank", fontsize=11)
    ax1.set_title("Layer 1 MLP eRank — Per Seed", fontsize=12)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Right panel: mean ± std summary ---
    x_mean = np.arange(n_metrics)
    means  = [vals.mean() for vals, _ in metrics.values()]
    stds   = [vals.std()  for vals, _ in metrics.values()]
    colors = [c for _, c in metrics.values()]

    bars2 = ax2.bar(x_mean, means, 0.55, yerr=stds, capsize=5,
                    color=colors, alpha=0.9, edgecolor="black", linewidth=0.5)

    for bar, m in zip(bars2, means):
        y_off = 0.3 if m >= 0 else -0.3
        va = "bottom" if m >= 0 else "top"
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + y_off,
                 f"{m:.2f}", ha="center", va=va, fontsize=10,
                 fontweight="bold")

    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xticks(x_mean)
    ax2.set_xticklabels(["$R_{\\mathrm{mid}}$", "$R_{\\mathrm{post}}$",
                          "$\\Delta_{\\mathrm{MLP}}$"], fontsize=10)
    ax2.set_title("Mean $\\pm$ std (3 seeds)", fontsize=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Standard Modular Addition — Layer 1 MLP Evidence",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "layer1_mlp_erank_evidence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
