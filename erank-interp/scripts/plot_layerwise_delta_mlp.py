#!/usr/bin/env python3
"""Bar chart: layerwise Δ_MLP eRank for modular addition (standard arch).

Shows Δ_MLP layer 0 < 0 and Δ_MLP layer 1 > 0, with error bars from 3 seeds.
"""

import json, pathlib, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED_DIR = pathlib.Path(__file__).resolve().parent.parent / "outputs" / "thesis_additions"
OUT_DIR  = SEED_DIR.parent / "advisor_core_figures"

def load_delta_mlp():
    """Return per-layer Δ_MLP values across seeds."""
    layer0, layer1 = [], []
    for seed in range(3):
        p = SEED_DIR / f"seed_{seed}" / "standard_modular_addition" / "erank" / "endpoint_erank.json"
        with open(p) as f:
            d = json.load(f)
        layer0.append(d["delta_erank_mlp"]["layer_0"])
        layer1.append(d["delta_erank_mlp"]["layer_1"])
    return np.array(layer0), np.array(layer1)

def main():
    l0, l1 = load_delta_mlp()
    means = [l0.mean(), l1.mean()]
    stds  = [l0.std(),  l1.std()]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(2)
    colors = ["#C44E52", "#E65100"]  # red-ish for L0, dark orange for L1
    bars = ax.bar(x, means, 0.55, yerr=stds, capsize=5,
                  color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)

    # Value annotations
    for bar, m, s in zip(bars, means, stds):
        offset = 0.25 if m >= 0 else -0.45
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{m:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["MLP Layer 0", "MLP Layer 1"], fontsize=12)
    ax.set_ylabel("Δ eRank  (mean ± std, 3 seeds)", fontsize=11)
    ax.set_title("Modular Addition — Layerwise Δ$_{\\mathrm{MLP}}$", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "layerwise_delta_mlp_modular_addition.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

if __name__ == "__main__":
    main()
