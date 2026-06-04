"""
scripts/aggregate_modadd_arch_matrix.py

Aggregator for the Part-A modular-addition architecture-order matrix.

Reads each run directory under outputs/modadd_arch_matrix_7gpu/ and writes:

    architecture_matrix_summary.csv          (run, schedule, train/val/test acc, tau)
    architecture_matrix_erank_summary.csv    (run, sum_d_attn, sum_d_mlp, layer-wise)
    plots/architecture_test_accuracy.png
    plots/architecture_train_val_curves.png
    plots/architecture_endpoint_erank_totals.png
    plots/architecture_grokking_erank_compression.png  (only grokked runs)
    plots/architecture_same_sum_ratio.png
    modadd_arch_matrix_report_for_chatgpt.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Canonical run order matches the launcher.
RUN_NAMES = [
    "modadd_4layer_standard_AMAMAMAM",
    "modadd_4layer_attention_only_AAAA",
    "modadd_4layer_mlp_only_MMMM",
    "modadd_4layer_early_attention_AMMM",
    "modadd_4layer_late_attention_MMMA",
    "modadd_4layer_route_then_process_AAMM",
    "modadd_4layer_process_then_route_MMAA",
]
SHORT_LABELS = {
    "modadd_4layer_standard_AMAMAMAM": "AMAMAMAM",
    "modadd_4layer_attention_only_AAAA": "AAAA",
    "modadd_4layer_mlp_only_MMMM": "MMMM",
    "modadd_4layer_early_attention_AMMM": "AMMM",
    "modadd_4layer_late_attention_MMMA": "MMMA",
    "modadd_4layer_route_then_process_AAMM": "AAMM",
    "modadd_4layer_process_then_route_MMAA": "MMAA",
}
GROKKING_THRESHOLD = 0.95


def _read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def _read_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _read_json(path: str) -> dict | list | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def collect_run(out_root: str, run_name: str) -> dict:
    rd = os.path.join(out_root, run_name)
    cfg = _read_yaml(os.path.join(rd, "config_used.yaml"))
    log_rows = _read_csv(os.path.join(rd, "training_log.csv"))
    erank_rows = _read_csv(os.path.join(rd, "endpoint_erank.csv"))
    erank_json = _read_json(os.path.join(rd, "endpoint_erank.json")) or {}
    spectrum_rows = _read_csv(os.path.join(rd, "spectrum_metrics.csv"))
    cluster_rows = _read_csv(os.path.join(rd, "same_sum_clustering.csv"))
    status = ""
    if os.path.exists(os.path.join(rd, "status.txt")):
        with open(os.path.join(rd, "status.txt")) as f:
            status = f.read().strip()

    # Find best-val checkpoint stats from log
    best_val_acc = -1.0
    best_val_epoch = None
    test_at_best_val = None
    tau = None
    for r in log_rows:
        ep = int(r["epoch"])
        va = _safe_float(r["val_acc"])
        ta = _safe_float(r["test_acc"])
        if va is None:
            continue
        if va > best_val_acc:
            best_val_acc = va
            best_val_epoch = ep
            test_at_best_val = ta
        if tau is None and va >= GROKKING_THRESHOLD:
            tau = ep

    # Summary eRank totals (the "sum_delta_attn" / "sum_delta_mlp" trailing rows)
    sum_da = sum_dm = None
    layer_erank: list[dict] = []
    for r in erank_rows:
        if "layer" in r and r.get("layer") not in (None, "", "sum_delta_attn", "sum_delta_mlp"):
            try:
                layer_erank.append({
                    "layer": int(r["layer"]),
                    "block_kind": r.get("block_kind", ""),
                    "erank_pre": _safe_float(r.get("erank_pre")),
                    "erank_mid": _safe_float(r.get("erank_mid")),
                    "erank_post": _safe_float(r.get("erank_post")),
                    "delta_attn": _safe_float(r.get("delta_attn")),
                    "delta_mlp": _safe_float(r.get("delta_mlp")),
                })
            except (ValueError, KeyError):
                pass
    # Trailing summary rows have layer == 'sum_delta_attn' / 'sum_delta_mlp'
    for r in erank_rows:
        if r.get("layer") == "sum_delta_attn":
            sum_da = _safe_float(r.get("block_kind"))
        if r.get("layer") == "sum_delta_mlp":
            sum_dm = _safe_float(r.get("block_kind"))

    final_erank = layer_erank[-1]["erank_post"] if layer_erank else None

    # Top-k mass
    topk = {}
    for r in spectrum_rows:
        try:
            topk[int(r["k"])] = float(r["topk_mass"])
        except (KeyError, ValueError):
            pass

    # Same-sum clustering
    cluster = None
    for r in cluster_rows:
        if r.get("metric") == "within_over_between_ratio":
            cluster = _safe_float(r.get("value"))

    return {
        "run_name": run_name,
        "schedule": "-".join(cfg.get("block_schedule", [])),
        "n_params": cfg.get("n_params"),
        "best_val_acc": best_val_acc if best_val_acc >= 0 else None,
        "best_val_epoch": best_val_epoch,
        "test_at_best_val": test_at_best_val,
        "tau": tau,
        "status": status,
        "log_rows": log_rows,
        "layer_erank": layer_erank,
        "sum_delta_attn": sum_da,
        "sum_delta_mlp": sum_dm,
        "final_erank": final_erank,
        "topk_mass": topk,
        "same_sum_ratio": cluster,
    }


def write_summary_csv(out_root: str, runs: list[dict]) -> None:
    p = os.path.join(out_root, "architecture_matrix_summary.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "run_name", "schedule", "n_params",
            "best_val_acc", "best_val_epoch", "test_at_best_val", "grokking_tau",
            "status",
        ])
        for r in runs:
            w.writerow([
                r["run_name"], r["schedule"], r["n_params"],
                r["best_val_acc"], r["best_val_epoch"], r["test_at_best_val"], r["tau"],
                r["status"],
            ])
    print(f"  wrote {p}")


def write_erank_summary_csv(out_root: str, runs: list[dict]) -> None:
    p = os.path.join(out_root, "architecture_matrix_erank_summary.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "run_name", "schedule",
            "final_erank", "sum_delta_attn", "sum_delta_mlp",
            "topk_mass_1", "topk_mass_5", "topk_mass_10", "topk_mass_20",
            "same_sum_ratio",
        ])
        for r in runs:
            tk = r.get("topk_mass") or {}
            w.writerow([
                r["run_name"], r["schedule"],
                r["final_erank"], r["sum_delta_attn"], r["sum_delta_mlp"],
                tk.get(1), tk.get(5), tk.get(10), tk.get(20),
                r["same_sum_ratio"],
            ])
    print(f"  wrote {p}")


def plot_test_accuracy(out_root: str, runs: list[dict]) -> None:
    plot_dir = os.path.join(out_root, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [SHORT_LABELS.get(r["run_name"], r["run_name"]) for r in runs]
    test_acc = [(r["test_at_best_val"] or 0.0) for r in runs]
    val_acc = [(r["best_val_acc"] or 0.0) for r in runs]
    x = np.arange(len(runs))
    width = 0.4
    ax.bar(x - width / 2, val_acc, width, label="best val_acc", color="steelblue")
    ax.bar(x + width / 2, test_acc, width, label="test @ best-val", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.axhline(GROKKING_THRESHOLD, color="red", linestyle=":", linewidth=1, label=f"grokking ({GROKKING_THRESHOLD})")
    ax.set_title("Modular-addition architecture matrix — accuracy")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(plot_dir, "architecture_test_accuracy.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")


def plot_train_val_curves(out_root: str, runs: list[dict]) -> None:
    plot_dir = os.path.join(out_root, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_t, ax_v = axes
    for r in runs:
        rows = r["log_rows"]
        if not rows:
            continue
        eps = [int(x["epoch"]) for x in rows]
        ta = [_safe_float(x["train_acc"]) or 0.0 for x in rows]
        va = [_safe_float(x["val_acc"]) or 0.0 for x in rows]
        label = SHORT_LABELS.get(r["run_name"], r["run_name"])
        ax_t.plot(eps, ta, label=label, alpha=0.85)
        ax_v.plot(eps, va, label=label, alpha=0.85)
    for ax, title in ((ax_t, "train_acc"), (ax_v, "val_acc")):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xscale("symlog", linthresh=10)
    fig.suptitle("Architecture matrix — training & validation curves")
    fig.tight_layout()
    p = os.path.join(plot_dir, "architecture_train_val_curves.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")


def plot_erank_totals(out_root: str, runs: list[dict]) -> None:
    plot_dir = os.path.join(out_root, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    labels = [SHORT_LABELS.get(r["run_name"], r["run_name"]) for r in runs]
    sda = [(r["sum_delta_attn"] or 0.0) for r in runs]
    sdm = [(r["sum_delta_mlp"] or 0.0) for r in runs]
    fer = [(r["final_erank"] or 0.0) for r in runs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(runs))
    width = 0.4
    ax1.bar(x - width / 2, sda, width, label="Σ Δ_attn", color="steelblue")
    ax1.bar(x + width / 2, sdm, width, label="Σ Δ_mlp", color="darkorange")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Sum of layer-wise eRank delta")
    ax1.set_title("Σ Δ_attn vs Σ Δ_mlp at best-val")
    ax1.legend()

    ax2.bar(x, fer, color="seagreen")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("eRank(R_final)")
    ax2.set_title("Final residual eRank at best-val")
    fig.tight_layout()
    p = os.path.join(plot_dir, "architecture_endpoint_erank_totals.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")


def plot_grokking_erank_compression(out_root: str, runs: list[dict]) -> None:
    plot_dir = os.path.join(out_root, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    grokked = [r for r in runs if r["best_val_acc"] is not None and r["best_val_acc"] >= GROKKING_THRESHOLD]
    if not grokked:
        print("  (no grokked runs — skipping compression plot)")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [SHORT_LABELS.get(r["run_name"], r["run_name"]) for r in grokked]
    final = [r["final_erank"] or 0.0 for r in grokked]
    ax.bar(np.arange(len(grokked)), final, color="seagreen")
    ax.set_xticks(np.arange(len(grokked)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("eRank(R_final)")
    ax.set_title("Grokked architectures — final residual eRank (lower = more compression)")
    fig.tight_layout()
    p = os.path.join(plot_dir, "architecture_grokking_erank_compression.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")


def plot_same_sum_ratio(out_root: str, runs: list[dict]) -> None:
    plot_dir = os.path.join(out_root, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    have = [r for r in runs if r["same_sum_ratio"] is not None]
    if not have:
        print("  (no same-sum ratios — skipping)")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [SHORT_LABELS.get(r["run_name"], r["run_name"]) for r in have]
    ratios = [r["same_sum_ratio"] for r in have]
    ax.bar(np.arange(len(have)), ratios, color="purple")
    ax.set_xticks(np.arange(len(have)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("within / between distance ratio")
    ax.set_title("Same-sum clustering ratio (lower = better class compactness)")
    fig.tight_layout()
    p = os.path.join(plot_dir, "architecture_same_sum_ratio.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")


def write_report(out_root: str, runs: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# Modular-addition Architecture Matrix Report")
    lines.append("")
    lines.append("## 1. Runs")
    lines.append("| run | schedule | params | best val | test@best | τ | status |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for r in runs:
        bv = "n/a" if r["best_val_acc"] is None else f"{r['best_val_acc']:.4f}"
        tb = "n/a" if r["test_at_best_val"] is None else f"{r['test_at_best_val']:.4f}"
        tau = "NA" if r["tau"] is None else f"{r['tau']}"
        params = "n/a" if r["n_params"] is None else f"{int(r['n_params']):,}"
        lines.append(
            f"| {SHORT_LABELS.get(r['run_name'], r['run_name'])} | "
            f"`{r['schedule']}` | {params} | {bv} | {tb} | {tau} | {r['status']} |"
        )
    lines.append("")
    lines.append("## 2. eRank summary (best-val)")
    lines.append("| run | final eRank | Σ Δ_attn | Σ Δ_mlp | top-1 | top-5 | top-10 | top-20 | same-sum ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in runs:
        tk = r.get("topk_mass") or {}
        def _f(x): return "n/a" if x is None else f"{x:.4f}"
        lines.append(
            f"| {SHORT_LABELS.get(r['run_name'], r['run_name'])} | "
            f"{_f(r['final_erank'])} | {_f(r['sum_delta_attn'])} | {_f(r['sum_delta_mlp'])} | "
            f"{_f(tk.get(1))} | {_f(tk.get(5))} | {_f(tk.get(10))} | {_f(tk.get(20))} | "
            f"{_f(r['same_sum_ratio'])} |"
        )
    lines.append("")
    lines.append("## 3. Per-layer eRank")
    for r in runs:
        lines.append(f"### {SHORT_LABELS.get(r['run_name'], r['run_name'])}  (`{r['schedule']}`)")
        lines.append("| layer | kind | pre | mid | post | Δ_attn | Δ_mlp |")
        lines.append("|---:|:--|---:|---:|---:|---:|---:|")
        for L in r["layer_erank"]:
            def _ff(x): return "NA" if x is None else f"{x:.3f}"
            lines.append(
                f"| {L['layer']} | {L['block_kind']} | {_ff(L['erank_pre'])} | "
                f"{_ff(L['erank_mid'])} | {_ff(L['erank_post'])} | "
                f"{_ff(L['delta_attn'])} | {_ff(L['delta_mlp'])} |"
            )
        lines.append("")
    lines.append("## 4. Plots")
    lines.append("- `plots/architecture_test_accuracy.png`")
    lines.append("- `plots/architecture_train_val_curves.png`")
    lines.append("- `plots/architecture_endpoint_erank_totals.png`")
    lines.append("- `plots/architecture_grokking_erank_compression.png` (grokked runs only)")
    lines.append("- `plots/architecture_same_sum_ratio.png`")
    lines.append("")
    p = os.path.join(out_root, "modadd_arch_matrix_report_for_chatgpt.md")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="outputs/modadd_arch_matrix_7gpu")
    args = parser.parse_args()
    out_root = os.path.join(PROJECT_ROOT, args.output_root)
    print(f"Aggregating from {out_root}")

    runs: list[dict] = []
    for name in RUN_NAMES:
        rd = os.path.join(out_root, name)
        if not os.path.isdir(rd):
            print(f"  (missing dir: {name})")
            continue
        runs.append(collect_run(out_root, name))

    if not runs:
        print("No runs found — nothing to aggregate.")
        return

    write_summary_csv(out_root, runs)
    write_erank_summary_csv(out_root, runs)
    plot_test_accuracy(out_root, runs)
    plot_train_val_curves(out_root, runs)
    plot_erank_totals(out_root, runs)
    plot_grokking_erank_compression(out_root, runs)
    plot_same_sum_ratio(out_root, runs)
    write_report(out_root, runs)
    print("Done.")


if __name__ == "__main__":
    main()
