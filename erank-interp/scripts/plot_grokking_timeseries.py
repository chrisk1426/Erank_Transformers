#!/usr/bin/env python3
"""Grokking time-series: validation accuracy + eRank(R_post layer 1) over epoch.

Dual-axis plot showing the co-evolution of generalization and representational
rank during grokking in modular addition (standard architecture).
"""

import pathlib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = (pathlib.Path(__file__).resolve().parent.parent /
            "outputs" / "control_rerun" / "training_time" /
            "standard_modular_addition_erank_over_time.csv")
OUT_DIR = (pathlib.Path(__file__).resolve().parent.parent /
           "outputs" / "advisor_core_figures")

COLOR_ERANK   = "#C44E52"   # consistent with MLP red
COLOR_VAL_ACC = "#4C72B0"   # consistent with existing blue palette


def main():
    df = pd.read_csv(CSV_PATH)

    # Extract layer-1, post-MLP rows
    l1_post = df[(df["layer"] == 1) & (df["hook"] == "post")].copy()
    l1_post = l1_post.sort_values("epoch")

    epochs   = l1_post["epoch"].values
    val_acc  = l1_post["val_acc"].values
    erank_l1 = l1_post["erank"].values

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    # eRank on left axis
    ln1 = ax1.plot(epochs, erank_l1, color=COLOR_ERANK, linewidth=2.2,
                   label="eRank($R_{\\mathrm{post}}^{(1)}$)", zorder=3)
    ax1.set_ylabel("eRank  ($R_{\\mathrm{post}}$, Layer 1)", fontsize=11,
                   color=COLOR_ERANK)
    ax1.tick_params(axis="y", labelcolor=COLOR_ERANK)

    # Val accuracy on right axis
    ln2 = ax2.plot(epochs, val_acc, color=COLOR_VAL_ACC, linewidth=2.2,
                   linestyle="--", label="Val accuracy", zorder=2)
    ax2.set_ylabel("Validation Accuracy", fontsize=11, color=COLOR_VAL_ACC)
    ax2.tick_params(axis="y", labelcolor=COLOR_VAL_ACC)
    ax2.set_ylim(-0.03, 1.08)

    # Shared x-axis
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_title("Modular Addition — Grokking Dynamics", fontsize=13)

    # Annotate grokking region
    grok_start = epochs[np.argmax(val_acc > 0.05)]  # first epoch above 5%
    grok_end   = epochs[np.argmax(val_acc > 0.99)]   # first epoch above 99%
    ax1.axvspan(grok_start, grok_end, alpha=0.10, color="gray", zorder=0)
    mid_grok = (grok_start + grok_end) / 2
    ax1.text(mid_grok, ax1.get_ylim()[1] * 0.95, "grokking\ntransition",
             ha="center", va="top", fontsize=9, color="gray", fontstyle="italic")

    # Mark grokking time τ_s  (first epoch where val_acc ≥ 0.99)
    tau_s = grok_end
    ax1.axvline(tau_s, color="#333333", linewidth=1.5, linestyle=":",
                zorder=4)
    # Place label above the line, near the top
    y_top = ax1.get_ylim()[1]
    ax1.annotate(
        f"$\\tau_s = {int(tau_s)}$",
        xy=(tau_s, y_top * 0.82), xytext=(tau_s - 800, y_top * 0.88),
        fontsize=11, fontweight="bold", color="#333333",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
        ha="center", va="bottom",
    )

    # Combined legend
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=10,
               framealpha=0.9)

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "grokking_timeseries_val_acc_erank_l1_post.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
