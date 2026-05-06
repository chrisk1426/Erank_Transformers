"""
scripts/generate_advisor_figures.py

Generate advisor-ready figures from completed seed-robust results.

Reads eRank JSON files and accuracy data directly from completed run outputs.
No model loading required.

Outputs:
  outputs/advisor_core_figures/core_accuracy_summary.png
  outputs/advisor_core_figures/core_erank_totals.png
  outputs/advisor_core_figures/modular_layerwise_erank.png
  outputs/advisor_core_figures/figure_inventory.md
"""

from __future__ import annotations

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = os.path.join(PROJECT_ROOT, "outputs", "thesis_additions")
OUT_DIR  = os.path.join(PROJECT_ROOT, "outputs", "advisor_core_figures")
SEEDS    = [0, 1, 2]

# Conditions with complete data (key_value and modular_addition, both archs)
CONDITIONS = [
    ("key_value",        "standard"),
    ("key_value",        "attention_only"),
    ("modular_addition", "standard"),
    ("modular_addition", "attention_only"),
]

TASK_LABELS = {
    "key_value":        "Key-Value\nRetrieval",
    "modular_addition": "Modular\nAddition",
}
ARCH_LABELS = {
    "standard":       "Standard",
    "attention_only": "Attn-Only",
}

# Colour scheme: standard=blue family, attention_only=orange family
COLORS = {
    ("key_value",        "standard"):       "#1565C0",
    ("key_value",        "attention_only"): "#42A5F5",
    ("modular_addition", "standard"):       "#E65100",
    ("modular_addition", "attention_only"): "#FFB74D",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_run_data() -> list[dict]:
    """Return one dict per (task, arch, seed) with accuracy and erank fields."""
    rows = []
    for task, arch in CONDITIONS:
        for seed in SEEDS:
            run_dir   = os.path.join(BASE_DIR, f"seed_{seed}", f"{arch}_{task}")
            curves_f  = os.path.join(run_dir, "results", "curves.json")
            erank_f   = os.path.join(run_dir, "erank", "endpoint_erank.json")

            row = {"task": task, "arch": arch, "seed": seed,
                   "test_acc": None, "val_acc": None,
                   "sum_da": None, "sum_dm": None,
                   "l0_da": None, "l0_dm": None,
                   "l1_da": None, "l1_dm": None}

            # Accuracy from curves.json
            if os.path.exists(curves_f):
                with open(curves_f) as f:
                    c = json.load(f)
                bve = c.get("best_val_epoch")
                row["val_acc"] = c.get("best_val_acc")
                if bve and bve in c.get("epoch", []):
                    idx = c["epoch"].index(bve)
                    row["test_acc"] = c["test_acc"][idx]
                else:
                    row["test_acc"] = c["test_acc"][-1] if c.get("test_acc") else None

            # eRank from endpoint_erank.json
            if os.path.exists(erank_f):
                with open(erank_f) as f:
                    e = json.load(f)
                da = e.get("delta_erank_attn", {})
                dm = e.get("delta_erank_mlp", {})
                row["sum_da"] = sum(da.values()) if da else 0.0
                row["sum_dm"] = sum(dm.values()) if dm else 0.0
                row["l0_da"]  = da.get("layer_0", 0.0)
                row["l0_dm"]  = dm.get("layer_0", 0.0)
                row["l1_da"]  = da.get("layer_1", 0.0)
                row["l1_dm"]  = dm.get("layer_1", 0.0)

            rows.append(row)
    return rows


def _agg(rows: list[dict], task: str, arch: str, field: str):
    vals = [r[field] for r in rows
            if r["task"] == task and r["arch"] == arch and r[field] is not None]
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


# ---------------------------------------------------------------------------
# Figure 1: Accuracy summary
# ---------------------------------------------------------------------------

def plot_accuracy_summary(rows: list[dict], out_dir: str) -> str:
    tasks = ["key_value", "modular_addition"]
    archs = ["standard", "attention_only"]

    fig, ax = plt.subplots(figsize=(7, 5))

    n_tasks = len(tasks)
    n_archs = len(archs)
    group_width = 0.7
    bar_width   = group_width / n_archs
    x = np.arange(n_tasks)

    for i, arch in enumerate(archs):
        means, stds = [], []
        for task in tasks:
            m, s = _agg(rows, task, arch, "test_acc")
            means.append(m)
            stds.append(s)
        offset = (i - (n_archs - 1) / 2) * bar_width
        color = COLORS[(tasks[0], arch)]  # use task-agnostic color per arch
        color = "#1565C0" if arch == "standard" else "#42A5F5"
        bars = ax.bar(x + offset, means, bar_width * 0.9,
                      yerr=stds, capsize=5,
                      color=color, alpha=0.9,
                      label=ARCH_LABELS[arch])
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=12)
    ax.set_ylabel("Test Accuracy (mean ± std, 3 seeds)", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.axhline(1/113, color="red", linestyle=":", linewidth=1.2, label="Chance (1/113)")
    ax.legend(fontsize=10)
    ax.set_title("Test Accuracy by Task and Architecture", fontsize=13, pad=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    path = os.path.join(out_dir, "core_accuracy_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 2: eRank totals
# ---------------------------------------------------------------------------

def plot_erank_totals(rows: list[dict], out_dir: str) -> str:
    tasks = ["key_value", "modular_addition"]
    archs = ["standard", "attention_only"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    components = [("sum_da", "Σ Δ_attn"), ("sum_dm", "Σ Δ_mlp")]

    for ax, (field, label) in zip(axes, components):
        n_tasks = len(tasks)
        n_archs = len(archs)
        bar_width = 0.32
        x = np.arange(n_tasks)

        for i, arch in enumerate(archs):
            means, stds = [], []
            for task in tasks:
                m, s = _agg(rows, task, arch, field)
                means.append(m)
                stds.append(s)
            offset = (i - (n_archs - 1) / 2) * bar_width
            color = "#1565C0" if arch == "standard" else "#42A5F5"
            bars = ax.bar(x + offset, means, bar_width * 0.9,
                          yerr=stds, capsize=4,
                          color=color, alpha=0.9,
                          label=ARCH_LABELS[arch])

        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=11)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(label, fontsize=12)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Note on attn-only modular: add annotation
    axes[0].annotate("attn-only mod.\n(failed runs)", xy=(1.18, 70), fontsize=8,
                     color="#42A5F5", ha="center")

    fig.suptitle("Endpoint eRank Deltas by Task and Architecture\n(mean ± std, 3 seeds)",
                 fontsize=13)
    fig.tight_layout()

    path = os.path.join(out_dir, "core_erank_totals.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 3: Modular addition layerwise eRank (standard only)
# ---------------------------------------------------------------------------

def plot_modular_layerwise(rows: list[dict], out_dir: str) -> str:
    task = "modular_addition"
    arch = "standard"

    mod_rows = [r for r in rows if r["task"] == task and r["arch"] == arch]
    fields_labels = [
        ("l0_da", "L0 Δ_attn", "#2196F3"),
        ("l0_dm", "L0 Δ_mlp",  "#FF9800"),
        ("l1_da", "L1 Δ_attn", "#1565C0"),
        ("l1_dm", "L1 Δ_mlp",  "#E65100"),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    n_fields = len(fields_labels)
    bar_width = 0.6
    x = np.arange(n_fields)

    means, stds = [], []
    for field, label, color in fields_labels:
        vals = [r[field] for r in mod_rows if r[field] is not None]
        means.append(float(np.mean(vals)) if vals else 0.0)
        stds.append(float(np.std(vals))   if vals else 0.0)

    colors_list = [fl[2] for fl in fields_labels]
    bars = ax.bar(x, means, bar_width, yerr=stds, capsize=5,
                  color=colors_list, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([fl[1] for fl in fields_labels], fontsize=12)
    ax.set_ylabel("Δ eRank (mean ± std, 3 seeds)", fontsize=11)
    ax.set_title("Modular Addition (Standard) — Layerwise eRank Deltas\n"
                 "L1 Δ_mlp > 0: late-layer MLP contribution", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.2 if m >= 0 else -0.5),
                f"{m:.2f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()

    path = os.path.join(out_dir, "modular_layerwise_erank.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure inventory report
# ---------------------------------------------------------------------------

def write_inventory(paths: list[tuple[str, str]], out_dir: str):
    lines = [
        "# Advisor Core Figures — Inventory",
        "",
        "All figures are from the completed seed-robust conditions:",
        "key_value (standard + attn-only) and modular_addition (standard + attn-only), 3 seeds each.",
        "",
        "| Figure | Path | Description |",
        "|---|---|---|",
    ]
    for name, path in paths:
        rel = os.path.relpath(path, PROJECT_ROOT)
        lines.append(f"| {name} | `{rel}` | See caption below |")

    lines += [
        "",
        "## Figure descriptions",
        "",
        "### core_accuracy_summary.png",
        "Bar chart of test accuracy (mean ± std, 3 seeds) for each task × architecture.",
        "Shows: key_value solved by both architectures; modular_addition solved only by standard.",
        "Red dotted line = chance level (1/113).",
        "",
        "### core_erank_totals.png",
        "Two bar charts: Σ Δ_attn (left) and Σ Δ_mlp (right) for each condition.",
        "Shows: key_value is attention-dominated (Σ Δ_attn > 0, Σ Δ_mlp < 0);",
        "modular_addition standard has positive Σ Δ_mlp.",
        "Note: attn-only modular_addition eRank is from failed runs (near-init checkpoint).",
        "",
        "### modular_layerwise_erank.png",
        "Per-layer breakdown for standard modular_addition.",
        "Shows that the positive MLP contribution is concentrated in Layer 1 (L1 Δ_mlp > 0).",
        "Layer 0 MLP contribution is near zero or negative.",
        "",
        "## Usage notes",
        "- Do not use attn-only modular_addition eRank bars in slides without noting the runs failed.",
        "- The hybrid task figures are not yet available (runs incomplete).",
    ]

    inv_path = os.path.join(out_dir, "figure_inventory.md")
    with open(inv_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {inv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading run data...")
    rows = _load_run_data()

    n_loaded = sum(1 for r in rows if r["test_acc"] is not None)
    print(f"Loaded {n_loaded}/{len(rows)} runs with accuracy data.")

    print("\nGenerating figures...")
    p1 = plot_accuracy_summary(rows, OUT_DIR)
    p2 = plot_erank_totals(rows, OUT_DIR)
    p3 = plot_modular_layerwise(rows, OUT_DIR)

    write_inventory([
        ("core_accuracy_summary.png",  p1),
        ("core_erank_totals.png",       p2),
        ("modular_layerwise_erank.png", p3),
    ], OUT_DIR)

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
