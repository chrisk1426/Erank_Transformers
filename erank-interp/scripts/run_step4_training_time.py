"""
Step 4: Training-time eRank analysis. Loads inline eRank data saved during
train_all_v2.py and generates time-series plots and report.
"""

from __future__ import annotations

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_ROOT  = os.path.join(PROJECT_ROOT, "outputs", "control_rerun")
RESULTS_DIR  = os.path.join(OUTPUT_ROOT, "results")
TT_DIR       = os.path.join(OUTPUT_ROOT, "training_time")

RUNS_OF_INTEREST = [
    "standard_modular_addition",
    "attention_only_modular_addition",
    "standard_key_value",
    "attention_only_key_value",
]

COLOR_PRE  = "#4C72B0"
COLOR_MID  = "#DD8452"
COLOR_POST = "#55A868"
COLOR_ATTN = "#4C72B0"
COLOR_MLP  = "#C44E52"
DPI = 150


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_erank_series(run_name: str) -> list[dict] | None:
    curves_path = os.path.join(RESULTS_DIR, f"{run_name}_curves.json")
    if not os.path.exists(curves_path):
        print(f"  WARNING: {curves_path} not found — skipping {run_name}")
        return None
    with open(curves_path) as f:
        history = json.load(f)
    series = history.get("erank_time_series", [])
    if not series:
        print(f"  WARNING: no erank_time_series in {curves_path} — skipping {run_name}")
        return None
    return series


def _infer_n_layers(series: list[dict]) -> int:
    entry = series[0]
    l = 0
    while f"layer_{l}_pre" in entry:
        l += 1
    return l


# ---------------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------------

def _save_csv(run_name: str, series: list[dict], n_layers: int) -> str:
    os.makedirs(TT_DIR, exist_ok=True)
    rows = []
    for entry in series:
        for l in range(n_layers):
            rows.append({
                "epoch":     entry["epoch"],
                "train_acc": entry["train_acc"],
                "val_acc":   entry["val_acc"],
                "test_acc":  entry.get("test_acc", ""),
                "layer":     l,
                "hook":      "pre",
                "erank":     entry.get(f"layer_{l}_pre", ""),
                "delta_attn": entry.get(f"layer_{l}_delta_attn", ""),
                "delta_mlp":  entry.get(f"layer_{l}_delta_mlp", ""),
            })
            rows.append({
                "epoch":     entry["epoch"],
                "train_acc": entry["train_acc"],
                "val_acc":   entry["val_acc"],
                "test_acc":  entry.get("test_acc", ""),
                "layer":     l,
                "hook":      "mid",
                "erank":     entry.get(f"layer_{l}_mid", ""),
                "delta_attn": "",
                "delta_mlp":  "",
            })
            rows.append({
                "epoch":     entry["epoch"],
                "train_acc": entry["train_acc"],
                "val_acc":   entry["val_acc"],
                "test_acc":  entry.get("test_acc", ""),
                "layer":     l,
                "hook":      "post",
                "erank":     entry.get(f"layer_{l}_post", ""),
                "delta_attn": "",
                "delta_mlp":  "",
            })

    csv_path = os.path.join(TT_DIR, f"{run_name}_erank_over_time.csv")
    fieldnames = ["epoch", "train_acc", "val_acc", "test_acc", "layer", "hook",
                  "erank", "delta_attn", "delta_mlp"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_erank_over_time(run_name: str, series: list[dict], n_layers: int) -> str:
    epochs     = [e["epoch"]    for e in series]
    val_accs   = [e["val_acc"]  for e in series]

    LAYER_COLORS = ["#4C72B0", "#DD8452"]
    HOOK_STYLES  = {"pre": "-", "mid": "--", "post": ":"}

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    for l in range(n_layers):
        col = LAYER_COLORS[l % len(LAYER_COLORS)]
        for hook, ls in HOOK_STYLES.items():
            vals = [e.get(f"layer_{l}_{hook}", np.nan) for e in series]
            ax1.plot(epochs, vals, linestyle=ls, color=col,
                     label=f"L{l} {hook}", linewidth=1.5)

    ax2.plot(epochs, val_accs, color="black", linewidth=2, linestyle="-.",
             alpha=0.7, label="val_acc")
    ax2.set_ylabel("val_acc", color="black")
    ax2.set_ylim(0, 1.05)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("eRank")
    ax1.set_title(f"{run_name}: eRank over training")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    fig.tight_layout()
    path = os.path.join(TT_DIR, f"{run_name}_erank_over_time.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def _plot_delta_over_time(run_name: str, series: list[dict], n_layers: int) -> str:
    epochs   = [e["epoch"]   for e in series]
    val_accs = [e["val_acc"] for e in series]

    LAYER_COLORS = ["#4C72B0", "#DD8452"]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    for l in range(n_layers):
        col = LAYER_COLORS[l % len(LAYER_COLORS)]
        da_vals = [e.get(f"layer_{l}_delta_attn", np.nan) for e in series]
        dm_vals = [e.get(f"layer_{l}_delta_mlp",  np.nan) for e in series]
        ax1.plot(epochs, da_vals, linestyle="-",  color=col,
                 label=f"L{l} Δ_attn", linewidth=1.5)
        ax1.plot(epochs, dm_vals, linestyle="--", color=col,
                 label=f"L{l} Δ_mlp",  linewidth=1.5)

    ax1.axhline(0, color="gray", linewidth=0.8, linestyle=":")

    ax2.plot(epochs, val_accs, color="black", linewidth=2, linestyle="-.",
             alpha=0.7, label="val_acc")
    ax2.set_ylabel("val_acc", color="black")
    ax2.set_ylim(0, 1.05)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("ΔeRank")
    ax1.set_title(f"{run_name}: Δ_attn / Δ_mlp over training")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    fig.tight_layout()
    path = os.path.join(TT_DIR, f"{run_name}_delta_over_time.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _find_first(series: list[dict], field: str, threshold: float) -> int | None:
    for e in series:
        if e.get(field, 0.0) >= threshold:
            return e["epoch"]
    return None


def _write_report(all_series: dict[str, list[dict]]) -> str:
    os.makedirs(TT_DIR, exist_ok=True)
    lines = ["# Training-Time eRank Report\n"]

    for run_name, series in all_series.items():
        if not series:
            continue

        n_layers = _infer_n_layers(series)
        epochs   = [e["epoch"]    for e in series]
        val_accs = [e["val_acc"]  for e in series]
        train_accs = [e["train_acc"] for e in series]

        mem_epoch = _find_first(series, "train_acc", 0.99)
        gen_epoch = _find_first(series, "val_acc",   0.50)

        lines.append(f"## {run_name}\n")
        lines.append(f"- eRank time-series points: {len(series)}")
        lines.append(f"- Epoch range: {epochs[0]} – {epochs[-1]}")
        lines.append(
            f"- First epoch with train_acc >= 99%: "
            f"{'epoch ' + str(mem_epoch) if mem_epoch else 'not reached in sampled epochs'}"
        )
        lines.append(
            f"- First epoch with val_acc >= 50%: "
            f"{'epoch ' + str(gen_epoch) if gen_epoch else 'not reached in sampled epochs'}"
        )

        # Compute total Δ_attn and Δ_mlp per entry.
        sum_da = [
            sum(e.get(f"layer_{l}_delta_attn", 0.0) for l in range(n_layers))
            for e in series
        ]
        sum_dm = [
            sum(e.get(f"layer_{l}_delta_mlp", 0.0) for l in range(n_layers))
            for e in series
        ]

        # Find where attn-vs-mlp dominance shifts.
        attn_dom = [(da > dm) for da, dm in zip(sum_da, sum_dm)]
        shifts = [
            epochs[i] for i in range(1, len(epochs)) if attn_dom[i] != attn_dom[i - 1]
        ]
        if shifts:
            lines.append(
                f"- Attention-vs-MLP dominance shift epochs: {shifts[:5]}"
            )
        else:
            lines.append(
                "- Attention-vs-MLP dominance: no shift detected in sampled epochs "
                f"({'attn dominant' if attn_dom[-1] else 'MLP dominant'} throughout)"
            )

        if "attention_only" in run_name:
            # Find if geometric signature precedes the accuracy peak.
            peak_val_acc = max(val_accs)
            peak_epoch   = epochs[val_accs.index(peak_val_acc)]
            # Look for a drop in sum_da before the peak.
            pre_peak = [(e, da) for e, da in zip(epochs, sum_da) if e < peak_epoch]
            if pre_peak:
                pre_epochs, pre_da = zip(*pre_peak)
                max_pre_da = max(pre_da)
                max_pre_ep = pre_epochs[list(pre_da).index(max_pre_da)]
                lines.append(
                    f"- Attention-only: peak val_acc={peak_val_acc:.4f} at epoch {peak_epoch}. "
                    f"Max Σ Δ_attn before peak: {max_pre_da:.3f} at epoch {max_pre_ep}."
                )
                if max_pre_ep < peak_epoch:
                    lines.append(
                        "  -> Geometric peak precedes accuracy peak, consistent with "
                        "attention reorganisation before collapse."
                    )
            else:
                lines.append(
                    f"- Attention-only: peak val_acc={peak_val_acc:.4f} at epoch {peak_epoch}. "
                    "No pre-peak geometry data available."
                )

        lines.append("")

    report_path = os.path.join(TT_DIR, "training_time_erank_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(TT_DIR, exist_ok=True)

    all_series: dict[str, list[dict]] = {}

    for run_name in RUNS_OF_INTEREST:
        print(f"\n{'='*50}")
        print(f"  {run_name}")
        print(f"{'='*50}")
        series = _load_erank_series(run_name)
        if series is None:
            continue

        n_layers = _infer_n_layers(series)
        print(f"  {len(series)} eRank checkpoints, {n_layers} layers")

        _save_csv(run_name, series, n_layers)
        _plot_erank_over_time(run_name, series, n_layers)
        _plot_delta_over_time(run_name, series, n_layers)

        all_series[run_name] = series

    _write_report(all_series)
    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()
