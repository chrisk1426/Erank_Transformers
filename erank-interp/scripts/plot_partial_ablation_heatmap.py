"""
scripts/plot_partial_ablation_heatmap.py

Stacked heatmap of partial-ablation dose-response for modular addition (top)
and key-value retrieval (bottom), standard transformer (3-seed mean test acc).

x-axis: ablation strength r  (output scaled by 1 - r)
y-axis: ablated component (L0/L1 x attn/MLP)
color : mean test accuracy

Reads:
  outputs/graded_ablation_complete/all_components_summary.csv

Writes:
  outputs/graded_ablation_complete/partial_ablation_heatmap.png
"""

from __future__ import annotations

import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = os.path.join(PROJECT_ROOT, "outputs", "graded_ablation_complete",
                        "all_components_summary.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "graded_ablation_complete",
                        "partial_ablation_heatmap.png")

COMPONENT_ORDER = [(0, "attn"), (0, "mlp"), (1, "attn"), (1, "mlp")]
COMPONENT_LABELS = ["L0 attention", "L0 MLP", "L1 attention", "L1 MLP"]
TASKS = [("modular_addition", "Modular addition"),
         ("key_value",        "Key-value retrieval")]


def _load_summary():
    rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            rows.append({
                "task": row["task"],
                "layer": int(row["layer"]),
                "component": row["component"],
                "r": float(row["r"]),
                "mean_acc": float(row["mean_acc"]),
            })
    return rows


def _matrix_for_task(rows, task):
    rs = sorted({r["r"] for r in rows if r["task"] == task})
    mat = np.full((len(COMPONENT_ORDER), len(rs)), np.nan)
    for i, (layer, comp) in enumerate(COMPONENT_ORDER):
        for j, r in enumerate(rs):
            match = [row for row in rows
                     if row["task"] == task and row["layer"] == layer
                     and row["component"] == comp and row["r"] == r]
            if match:
                mat[i, j] = match[0]["mean_acc"]
    return mat, rs


def main():
    rows = _load_summary()

    fig, axes = plt.subplots(2, 1, figsize=(9, 6),
                             gridspec_kw={"hspace": 0.30, "right": 0.88})
    cmap = "viridis"
    vmin, vmax = 0.0, 1.0

    last_im = None
    for ax, (task_key, task_label) in zip(axes, TASKS):
        mat, rs = _matrix_for_task(rows, task_key)
        last_im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                            origin="upper")
        ax.set_xticks(np.arange(len(rs)))
        ax.set_xticklabels([f"{r:.2f}" for r in rs], fontsize=9)
        ax.set_yticks(np.arange(len(COMPONENT_ORDER)))
        ax.set_yticklabels(COMPONENT_LABELS, fontsize=10)
        ax.set_xlabel("Ablation strength r  (output scaled by 1 − r)", fontsize=10)
        ax.set_title(task_label, fontsize=11, pad=4)

        # Annotate each cell with the accuracy value.
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isnan(val):
                    continue
                # Use white text on dark cells, black on light cells.
                txt_color = "white" if val < 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=txt_color)

    # Shared colorbar on the right.
    cbar_ax = fig.add_axes([0.90, 0.12, 0.020, 0.76])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("Test accuracy (3-seed mean)", fontsize=10)

    fig.suptitle("Partial ablation dose-response — standard transformer (3 seeds)",
                 fontsize=12, y=0.985)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
