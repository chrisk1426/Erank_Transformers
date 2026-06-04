"""
scripts/analyze_layer0_freeze_modadd.py

Aggregator + report builder for the layer-0 freezing modular-addition experiment.

Produces (under <output_root>/):
    layer0_freeze_summary.csv
    layer0_freeze_erank_summary.csv
    layer0_freeze_grokking_summary.csv
    plots/
        accuracy_curves_<variant>.png
        final_rep_erank_over_time_by_variant.png
        mlp_delta_by_layer_over_time_<variant>.png
        attn_delta_by_layer_over_time_<variant>.png
        best_val_layerwise_erank_deltas_mean_std.png
        grokking_vs_final_erank.png
        grokking_time_and_test_acc_by_variant.png
        same_sum_ratio_over_time_by_variant.png       (if same_sum data available)
    layer0_freeze_modadd_results_for_chatgpt.md
    layer0_freeze_modadd_results_for_chatgpt.json

Tolerates partial state: writes interim if any (variant, seed) is missing.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

VARIANTS = ["normal", "freeze_A0", "freeze_M0", "freeze_A0_M0"]
SEEDS = [0, 1, 2]
DONE_STATUSES = {"COMPLETED", "GROKKED", "MEMORIZED_ONLY", "FAILED_TO_FIT", "PARTIAL"}
VARIANT_COLOR = {
    "normal":      "tab:blue",
    "freeze_A0":   "tab:orange",
    "freeze_M0":   "tab:green",
    "freeze_A0_M0":"tab:red",
}


def safe_float(x):
    if x in (None, "", "NA"):
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def safe_int(x):
    if x in (None, "", "NA"):
        return None
    try:
        return int(x)
    except Exception:
        return None


def load_run(out_root: str, run_name: str) -> dict | None:
    run_dir = os.path.join(out_root, run_name)
    if not os.path.isdir(run_dir):
        return None
    out: dict = {"run_dir": run_dir, "run_name": run_name}
    sp = os.path.join(run_dir, "status.txt")
    if os.path.exists(sp):
        out["status_file"] = open(sp).read().strip().upper()
    js = os.path.join(run_dir, "run_summary.json")
    if os.path.exists(js):
        out["summary"] = json.load(open(js))
    for csvname, key in [("training_log.csv", "training_log"),
                         ("erank_by_checkpoint.csv", "erank_by_ckpt"),
                         ("erank_summary_by_checkpoint.csv", "erank_summary"),
                         ("same_sum_clustering.csv", "same_sum"),
                         ("checkpoint_manifest.csv", "ckpt_manifest")]:
        p = os.path.join(run_dir, csvname)
        if os.path.exists(p):
            with open(p) as f:
                out[key] = list(csv.DictReader(f))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/layer0_freeze_modadd")
    args = parser.parse_args()

    out_root = os.path.join(PROJECT_ROOT, args.output_root) \
        if not os.path.isabs(args.output_root) else args.output_root
    plot_dir = os.path.join(out_root, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    runs: dict[str, dict] = {}
    for v in VARIANTS:
        for s in SEEDS:
            run_name = f"{v}_seed{s}"
            r = load_run(out_root, run_name)
            runs[run_name] = r if r is not None else {"run_name": run_name}

    # ----- layer0_freeze_summary.csv -----
    rows = []
    for v in VARIANTS:
        for s in SEEDS:
            run_name = f"{v}_seed{s}"
            r = runs.get(run_name) or {}
            sumdat = r.get("summary") or {}
            rows.append({
                "variant": v, "seed": s,
                "freeze_A0": int(sumdat.get("freeze_A0", v in ("freeze_A0","freeze_A0_M0"))),
                "freeze_M0": int(sumdat.get("freeze_M0", v in ("freeze_M0","freeze_A0_M0"))),
                "best_val_epoch": sumdat.get("best_val_epoch", ""),
                "best_val_acc":   sumdat.get("best_val_acc", ""),
                "test_acc_at_best_val": sumdat.get("test_acc_at_best_val", ""),
                "final_train_acc": sumdat.get("final_train_acc", ""),
                "final_val_acc":   sumdat.get("final_val_acc", ""),
                "tau_0.90":        sumdat.get("tau_0.90", ""),
                "tau_0.95":        sumdat.get("tau_0.95", ""),
                "status": sumdat.get("status", r.get("status_file", "PENDING")),
            })
    summary_fields = ["variant","seed","freeze_A0","freeze_M0",
                      "best_val_epoch","best_val_acc","test_acc_at_best_val",
                      "final_train_acc","final_val_acc","tau_0.90","tau_0.95","status"]
    with open(os.path.join(out_root, "layer0_freeze_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields); w.writeheader()
        for r in rows: w.writerow(r)

    # ----- layer0_freeze_erank_summary.csv -----
    erank_rows: list[dict] = []
    for v in VARIANTS:
        for s in SEEDS:
            run = runs.get(f"{v}_seed{s}")
            if not run or "erank_summary" not in run:
                continue
            for r in run["erank_summary"]:
                er = dict(r)
                # Identify strongest negative MLP delta layer.
                deltas = {0: safe_float(er.get("layer0_delta_mlp")),
                          1: safe_float(er.get("layer1_delta_mlp")),
                          2: safe_float(er.get("layer2_delta_mlp"))}
                neg = {l: d for l, d in deltas.items() if not math.isnan(d)}
                if neg:
                    most_neg_l = min(neg, key=neg.get)
                    er["strongest_negative_mlp_layer"] = most_neg_l
                    er["strongest_negative_mlp_delta"] = neg[most_neg_l]
                else:
                    er["strongest_negative_mlp_layer"] = ""
                    er["strongest_negative_mlp_delta"] = ""
                erank_rows.append(er)
    erank_fields = ["run_name","variant","seed","epoch","checkpoint_label",
                    "train_acc","val_acc","sum_delta_attn","sum_delta_mlp",
                    "final_rep_erank",
                    "layer0_delta_attn","layer1_delta_attn","layer2_delta_attn",
                    "layer0_delta_mlp","layer1_delta_mlp","layer2_delta_mlp",
                    "strongest_negative_mlp_layer","strongest_negative_mlp_delta"]
    with open(os.path.join(out_root, "layer0_freeze_erank_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=erank_fields, extrasaction="ignore")
        w.writeheader()
        for r in erank_rows:
            w.writerow(r)

    # ----- layer0_freeze_grokking_summary.csv -----
    agg_rows: list[dict] = []
    aggregate_for_md: dict[str, dict] = {}
    for v in VARIANTS:
        bv = [safe_float(r["best_val_acc"]) for r in rows if r["variant"] == v]
        tv = [safe_float(r["test_acc_at_best_val"]) for r in rows if r["variant"] == v]
        t90 = [safe_float(r["tau_0.90"]) for r in rows if r["variant"] == v]
        t95 = [safe_float(r["tau_0.95"]) for r in rows if r["variant"] == v]
        statuses = [r["status"] for r in rows if r["variant"] == v]

        # best-val per-layer mlp deltas from erank_rows (checkpoint_label == best_val).
        bv_mlp = {0: [], 1: [], 2: []}
        for r in erank_rows:
            if r.get("variant") != v: continue
            if r.get("checkpoint_label") != "best_val": continue
            for l in (0, 1, 2):
                d = safe_float(r.get(f"layer{l}_delta_mlp"))
                if not math.isnan(d):
                    bv_mlp[l].append(d)

        n = len(bv)
        def _safe_mean(xs):
            xs = [x for x in xs if not math.isnan(x)]
            return float(np.mean(xs)) if xs else float("nan")
        def _safe_std(xs):
            xs = [x for x in xs if not math.isnan(x)]
            return float(np.std(xs)) if xs else float("nan")
        agg = {
            "variant": v, "n_seeds": n,
            "mean_best_val_acc": _safe_mean(bv),
            "std_best_val_acc":  _safe_std(bv),
            "mean_test_acc_at_best_val": _safe_mean(tv),
            "std_test_acc_at_best_val":  _safe_std(tv),
            "mean_tau_0.90": _safe_mean(t90),
            "std_tau_0.90":  _safe_std(t90),
            "mean_tau_0.95": _safe_mean(t95),
            "std_tau_0.95":  _safe_std(t95),
            "fraction_grokked_0.90":
                (sum(1 for x in bv if not math.isnan(x) and x >= 0.90) / n) if n else float("nan"),
            "fraction_grokked_0.95":
                (sum(1 for x in bv if not math.isnan(x) and x >= 0.95) / n) if n else float("nan"),
            "mean_layer0_delta_mlp_best_val": _safe_mean(bv_mlp[0]),
            "std_layer0_delta_mlp_best_val":  _safe_std(bv_mlp[0]),
            "mean_layer1_delta_mlp_best_val": _safe_mean(bv_mlp[1]),
            "std_layer1_delta_mlp_best_val":  _safe_std(bv_mlp[1]),
            "mean_layer2_delta_mlp_best_val": _safe_mean(bv_mlp[2]),
            "std_layer2_delta_mlp_best_val":  _safe_std(bv_mlp[2]),
        }
        agg_rows.append(agg)
        aggregate_for_md[v] = agg

    agg_fields = list(agg_rows[0].keys()) if agg_rows else ["variant"]
    with open(os.path.join(out_root, "layer0_freeze_grokking_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields); w.writeheader()
        for r in agg_rows: w.writerow(r)

    # ----- Plots -----
    plot_paths: dict[str, str] = {}

    # Plot 1: per-variant accuracy curves (one plot per variant).
    for v in VARIANTS:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for s in SEEDS:
            run = runs.get(f"{v}_seed{s}")
            if not run or "training_log" not in run: continue
            ep = [int(r["epoch"]) for r in run["training_log"]]
            tr = [safe_float(r["train_acc"]) for r in run["training_log"]]
            va = [safe_float(r["val_acc"]) for r in run["training_log"]]
            ax.plot(ep, tr, alpha=0.5, linestyle="--", linewidth=0.9, label=f"seed{s} train")
            ax.plot(ep, va, linewidth=1.3, label=f"seed{s} val")
        ax.axhline(0.90, color="purple", linestyle="--", linewidth=0.6, label="0.90")
        ax.axhline(0.95, color="purple", linestyle="-",  linewidth=0.6, label="0.95")
        ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"Accuracy curves: {v}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = os.path.join(plot_dir, f"accuracy_curves_{v}.png")
        fig.savefig(p, dpi=140); plt.close(fig)
        plot_paths[f"accuracy_curves_{v}"] = p

    # Plot 2: final residual eRank over training, by variant (mean ± seed curves).
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for v in VARIANTS:
        for s in SEEDS:
            run = runs.get(f"{v}_seed{s}")
            if not run or "erank_summary" not in run: continue
            er = [r for r in run["erank_summary"]
                  if r.get("checkpoint_label") in ("periodic_full",)
                  and not math.isnan(safe_float(r.get("epoch")))]
            er = sorted(er, key=lambda r: safe_int(r["epoch"]) or 0)
            if not er: continue
            ep = [safe_int(r["epoch"]) for r in er]
            ee = [safe_float(r["final_rep_erank"]) for r in er]
            ax.plot(ep, ee, color=VARIANT_COLOR[v], alpha=0.35, linewidth=0.9)
    # mean curve per variant — coarse-grained on epochs that appear consistently.
    for v in VARIANTS:
        all_pts: dict[int, list[float]] = defaultdict(list)
        for s in SEEDS:
            run = runs.get(f"{v}_seed{s}")
            if not run or "erank_summary" not in run: continue
            for r in run["erank_summary"]:
                if r.get("checkpoint_label") != "periodic_full": continue
                ep = safe_int(r["epoch"]); er = safe_float(r["final_rep_erank"])
                if ep is None or math.isnan(er): continue
                all_pts[ep].append(er)
        if all_pts:
            sorted_eps = sorted(all_pts.keys())
            means = [float(np.mean(all_pts[e])) for e in sorted_eps]
            ax.plot(sorted_eps, means, color=VARIANT_COLOR[v], linewidth=2.0,
                    label=f"{v} (mean over seeds)")
    ax.set_xlabel("epoch"); ax.set_ylabel("final-rep eRank")
    ax.set_title("Final residual eRank over training, by variant")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(plot_dir, "final_rep_erank_over_time_by_variant.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    plot_paths["final_rep_erank_over_time_by_variant"] = p

    # Plot 3: layerwise MLP delta over time per variant.
    for v in VARIANTS:
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        for s in SEEDS:
            run = runs.get(f"{v}_seed{s}")
            if not run or "erank_summary" not in run: continue
            er = sorted([r for r in run["erank_summary"]
                         if r.get("checkpoint_label") == "periodic_full"],
                        key=lambda r: safe_int(r["epoch"]) or 0)
            ep = [safe_int(r["epoch"]) for r in er]
            for l, color in [(0,"tab:blue"),(1,"tab:orange"),(2,"tab:green")]:
                vals = [safe_float(r[f"layer{l}_delta_mlp"]) for r in er]
                ax.plot(ep, vals, color=color, alpha=0.6, linewidth=1.0,
                        label=f"seed{s} layer{l}" if s == 0 else None)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("epoch"); ax.set_ylabel("Δ_MLP per layer")
        ax.set_title(f"Layerwise MLP delta over training: {v}")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = os.path.join(plot_dir, f"mlp_delta_by_layer_over_time_{v}.png")
        fig.savefig(p, dpi=140); plt.close(fig)
        plot_paths[f"mlp_delta_by_layer_over_time_{v}"] = p

    # Plot 4: layerwise attention delta over time per variant.
    for v in VARIANTS:
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        for s in SEEDS:
            run = runs.get(f"{v}_seed{s}")
            if not run or "erank_summary" not in run: continue
            er = sorted([r for r in run["erank_summary"]
                         if r.get("checkpoint_label") == "periodic_full"],
                        key=lambda r: safe_int(r["epoch"]) or 0)
            ep = [safe_int(r["epoch"]) for r in er]
            for l, color in [(0,"tab:blue"),(1,"tab:orange"),(2,"tab:green")]:
                vals = [safe_float(r[f"layer{l}_delta_attn"]) for r in er]
                ax.plot(ep, vals, color=color, alpha=0.6, linewidth=1.0,
                        label=f"seed{s} layer{l}" if s == 0 else None)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("epoch"); ax.set_ylabel("Δ_attn per layer")
        ax.set_title(f"Layerwise attention delta over training: {v}")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = os.path.join(plot_dir, f"attn_delta_by_layer_over_time_{v}.png")
        fig.savefig(p, dpi=140); plt.close(fig)
        plot_paths[f"attn_delta_by_layer_over_time_{v}"] = p

    # Plot 5: best-val layerwise delta bar chart (mean ± std across seeds).
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(6)  # 3 layers × 2 (attn,mlp)
    width = 0.18
    for vi, v in enumerate(VARIANTS):
        means = []; stds = []
        for l in (0, 1, 2):
            for k in ("delta_attn", "delta_mlp"):
                vals = []
                for s in SEEDS:
                    run = runs.get(f"{v}_seed{s}")
                    if not run or "erank_summary" not in run: continue
                    for r in run["erank_summary"]:
                        if r.get("checkpoint_label") != "best_val": continue
                        col = f"layer{l}_{k}"
                        d = safe_float(r.get(col))
                        if not math.isnan(d):
                            vals.append(d); break
                means.append(float(np.mean(vals)) if vals else 0.0)
                stds.append(float(np.std(vals)) if vals else 0.0)
        ax.bar(x + (vi - 1.5) * width, means, width=width, yerr=stds,
               color=VARIANT_COLOR[v], capsize=2, label=v,
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(["L0 Δa","L0 Δm","L1 Δa","L1 Δm","L2 Δa","L2 Δm"])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Δ eRank (mean ± std across seeds)")
    ax.set_title("Best-val layerwise eRank deltas by variant")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(plot_dir, "best_val_layerwise_erank_deltas_mean_std.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    plot_paths["best_val_layerwise_erank_deltas_mean_std"] = p

    # Plot 6: grokking vs final eRank (val_acc & final_rep_erank vs epoch, per variant).
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ai, v in enumerate(VARIANTS):
        ax = axes[ai // 2, ai % 2]
        ax2 = ax.twinx()
        for s in SEEDS:
            run = runs.get(f"{v}_seed{s}")
            if not run: continue
            if "training_log" in run:
                ep_t = [int(r["epoch"]) for r in run["training_log"]]
                va = [safe_float(r["val_acc"]) for r in run["training_log"]]
                ax.plot(ep_t, va, color="tab:blue", alpha=0.5, linewidth=1.0)
            if "erank_summary" in run:
                er = sorted([r for r in run["erank_summary"]
                             if r.get("checkpoint_label") == "periodic_full"],
                            key=lambda r: safe_int(r["epoch"]) or 0)
                ep_e = [safe_int(r["epoch"]) for r in er]
                ee = [safe_float(r["final_rep_erank"]) for r in er]
                ax2.plot(ep_e, ee, color="tab:red", alpha=0.5, linewidth=1.0,
                          linestyle="-.")
        ax.set_title(v)
        ax.set_ylabel("val_acc (blue)", color="tab:blue")
        ax2.set_ylabel("final_rep_eRank (red)", color="tab:red")
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(0.90, color="purple", linestyle="--", linewidth=0.4)
        ax.axhline(0.95, color="purple", linestyle="-",  linewidth=0.4)
        ax.grid(True, alpha=0.3)
        if ai // 2 == 1:
            ax.set_xlabel("epoch")
    fig.suptitle("Grokking vs final-rep eRank")
    fig.tight_layout()
    p = os.path.join(plot_dir, "grokking_vs_final_erank.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    plot_paths["grokking_vs_final_erank"] = p

    # Plot 7: tau and test_acc by variant (mean ± std).
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [("mean_tau_0.90","std_tau_0.90", "τ at val=0.90"),
               ("mean_tau_0.95","std_tau_0.95", "τ at val=0.95"),
               ("mean_test_acc_at_best_val","std_test_acc_at_best_val", "test acc @ best-val")]
    for ai, (mkey, skey, title) in enumerate(metrics):
        ax = axes[ai]
        xs = np.arange(len(VARIANTS))
        means = [aggregate_for_md[v][mkey] for v in VARIANTS]
        stds  = [aggregate_for_md[v][skey] for v in VARIANTS]
        bars = ax.bar(xs, means, yerr=stds, color=[VARIANT_COLOR[v] for v in VARIANTS],
                      edgecolor="black", linewidth=0.4, capsize=3)
        ax.set_xticks(xs); ax.set_xticklabels(VARIANTS, rotation=15, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        for b, m in zip(bars, means):
            if not (m is None or math.isnan(m)):
                ax.annotate(f"{m:.3f}" if "acc" in title else f"{m:.0f}",
                            (b.get_x()+b.get_width()/2, m), ha="center", fontsize=7,
                            va="bottom")
    fig.tight_layout()
    p = os.path.join(plot_dir, "grokking_time_and_test_acc_by_variant.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    plot_paths["grokking_time_and_test_acc_by_variant"] = p

    # Plot 8 (optional): same-sum ratio over time by variant.
    have_ss = any("same_sum" in (runs.get(f"{v}_seed{s}") or {}) for v in VARIANTS for s in SEEDS)
    if have_ss:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for v in VARIANTS:
            for s in SEEDS:
                run = runs.get(f"{v}_seed{s}")
                if not run or "same_sum" not in run: continue
                ss = sorted(run["same_sum"], key=lambda r: safe_int(r["epoch"]) or 0)
                ep = [safe_int(r["epoch"]) for r in ss]
                ratio = [safe_float(r["within_between_ratio"]) for r in ss]
                ax.plot(ep, ratio, color=VARIANT_COLOR[v], alpha=0.4, linewidth=0.9,
                        label=v if s == 0 else None, marker="o", markersize=3)
        ax.set_xlabel("epoch"); ax.set_ylabel("D_within / D_between")
        ax.set_title("Same-sum clustering ratio over time, by variant")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = os.path.join(plot_dir, "same_sum_ratio_over_time_by_variant.png")
        fig.savefig(p, dpi=140); plt.close(fig)
        plot_paths["same_sum_ratio_over_time_by_variant"] = p

    # ----- Markdown report -----
    md_lines: list[str] = []
    md_lines.append("# Layer-0 Freezing Modular-Addition Results")
    md_lines.append("")
    md_lines.append(f"timestamp: {dt.datetime.now().isoformat(timespec='seconds')}")
    md_lines.append("")
    md_lines.append("## 1. Executive summary")
    md_lines.append("")
    summary_lines = []
    for v in VARIANTS:
        a = aggregate_for_md[v]
        summary_lines.append(
            f"- **{v}**: mean test@best-val = {a['mean_test_acc_at_best_val']:.4f} ± "
            f"{a['std_test_acc_at_best_val']:.4f}; "
            f"τ_0.95 = {a['mean_tau_0.95']:.1f} ± {a['std_tau_0.95']:.1f}; "
            f"fraction grokked (0.95) = {a['fraction_grokked_0.95']:.2f}"
        )
    md_lines.extend(summary_lines)
    md_lines.append("")
    md_lines.append("## 2. Experimental design")
    md_lines.append("")
    md_lines.append("- 3-layer HookedTransformer, d_model=128, 4 heads, d_mlp=512, GELU MLPs")
    md_lines.append("- Task: modular addition c = (a + b) mod 113")
    md_lines.append("- AdamW lr=3e-4, wd=1.0, batch=256, max_epochs=20000")
    md_lines.append("- 4 variants × 3 seeds = 12 runs. Shared base init per seed.")
    md_lines.append("- Frozen components are kept in the forward pass but `requires_grad=False`.")
    md_lines.append("- Grokking thresholds τ_0.90 and τ_0.95 use validation accuracy, not test.")
    md_lines.append("- Test accuracy is evaluated only at the best-validation checkpoint.")
    md_lines.append("")
    md_lines.append("## 3. Trainability verification")
    md_lines.append("")
    md_lines.append("| variant | A0 frozen? | M0 frozen? |")
    md_lines.append("|:--|:-:|:-:|")
    md_lines.append("| normal       | no  | no  |")
    md_lines.append("| freeze_A0    | yes | no  |")
    md_lines.append("| freeze_M0    | no  | yes |")
    md_lines.append("| freeze_A0_M0 | yes | yes |")
    md_lines.append("")
    md_lines.append("(Per-run `trainability_summary.txt` for parameter counts.)")
    md_lines.append("")
    md_lines.append("## 4. Accuracy and grokking results")
    md_lines.append("")
    md_lines.append("Per (variant, seed):")
    md_lines.append("")
    md_lines.append("| variant | seed | best_val_epoch | best_val_acc | test_acc_at_best_val | final_train_acc | final_val_acc | τ_0.90 | τ_0.95 | status |")
    md_lines.append("|:--|---:|---:|---:|---:|---:|---:|---:|---:|:--|")
    def fmt(x, p=4):
        if x in ("", None): return "NA"
        try: return f"{float(x):.{p}f}"
        except: return str(x)
    for r in rows:
        md_lines.append(
            f"| {r['variant']} | {r['seed']} | {r['best_val_epoch']} | "
            f"{fmt(r['best_val_acc'])} | {fmt(r['test_acc_at_best_val'])} | "
            f"{fmt(r['final_train_acc'])} | {fmt(r['final_val_acc'])} | "
            f"{r['tau_0.90']} | {r['tau_0.95']} | {r['status']} |"
        )
    md_lines.append("")
    md_lines.append("Aggregate across seeds:")
    md_lines.append("")
    md_lines.append("| variant | mean test acc | std test acc | frac grok 0.90 | mean τ_0.90 | frac grok 0.95 | mean τ_0.95 |")
    md_lines.append("|:--|---:|---:|---:|---:|---:|---:|")
    for v in VARIANTS:
        a = aggregate_for_md[v]
        md_lines.append(
            f"| {v} | {a['mean_test_acc_at_best_val']:.4f} | {a['std_test_acc_at_best_val']:.4f} | "
            f"{a['fraction_grokked_0.90']:.2f} | {a['mean_tau_0.90']:.0f} | "
            f"{a['fraction_grokked_0.95']:.2f} | {a['mean_tau_0.95']:.0f} |"
        )
    md_lines.append("")
    md_lines.append("## 5. eRank summary at best-val")
    md_lines.append("")
    md_lines.append("| variant | layer | mean Δ_attn | std Δ_attn | mean Δ_mlp | std Δ_mlp |")
    md_lines.append("|:--|---:|---:|---:|---:|---:|")
    for v in VARIANTS:
        for l in (0, 1, 2):
            vals_a, vals_m = [], []
            for s in SEEDS:
                run = runs.get(f"{v}_seed{s}")
                if not run or "erank_summary" not in run: continue
                for r in run["erank_summary"]:
                    if r.get("checkpoint_label") != "best_val": continue
                    da = safe_float(r.get(f"layer{l}_delta_attn"))
                    dm = safe_float(r.get(f"layer{l}_delta_mlp"))
                    if not math.isnan(da): vals_a.append(da)
                    if not math.isnan(dm): vals_m.append(dm)
                    break
            md_lines.append(
                f"| {v} | {l} | "
                f"{(np.mean(vals_a) if vals_a else float('nan')):.3f} | "
                f"{(np.std(vals_a)  if vals_a else float('nan')):.3f} | "
                f"{(np.mean(vals_m) if vals_m else float('nan')):.3f} | "
                f"{(np.std(vals_m)  if vals_m else float('nan')):.3f} |"
            )
    md_lines.append("")
    md_lines.append("## 6. Training-time eRank dynamics")
    md_lines.append("")
    md_lines.append("Plots:")
    md_lines.append("- `plots/final_rep_erank_over_time_by_variant.png`")
    for v in VARIANTS:
        md_lines.append(f"- `plots/mlp_delta_by_layer_over_time_{v}.png`")
        md_lines.append(f"- `plots/attn_delta_by_layer_over_time_{v}.png`")
    md_lines.append("- `plots/grokking_vs_final_erank.png`")
    md_lines.append("- `plots/best_val_layerwise_erank_deltas_mean_std.png`")
    md_lines.append("- `plots/grokking_time_and_test_acc_by_variant.png`")
    if have_ss:
        md_lines.append("- `plots/same_sum_ratio_over_time_by_variant.png`")
    md_lines.append("")
    md_lines.append("## 7. Compression analysis")
    md_lines.append("")
    md_lines.append("For each variant, the layer with the strongest negative Δ_MLP at best-val is the candidate compressive layer. See `layer0_freeze_erank_summary.csv` (`strongest_negative_mlp_layer`) for per-run identification.")
    md_lines.append("")
    md_lines.append("## 8. Same-sum clustering")
    md_lines.append("")
    if have_ss:
        md_lines.append("See `plots/same_sum_ratio_over_time_by_variant.png` and per-run `same_sum_clustering.csv`.")
    else:
        md_lines.append("Same-sum clustering data not available for any run.")
    md_lines.append("")
    md_lines.append("## 9. Interpretation")
    md_lines.append("")
    md_lines.append("(Fill in based on actual results: necessity of trainable M0 compression, whether compression shifts to M1/M2 when M0 is frozen, whether frozen A0 disrupts routing, etc.)")
    md_lines.append("")
    md_lines.append("## 10. Caveats")
    md_lines.append("")
    md_lines.append("- Only 3 seeds.")
    md_lines.append("- Frozen ≠ ablated: frozen components remain in the forward pass.")
    md_lines.append("- Test accuracy was used only for final evaluation, not for grokking detection or checkpoint selection.")
    md_lines.append("- Dense checkpointing is storage-heavy; model-only ckpts every 10 epochs, full every 100.")
    md_lines.append("- eRank is geometric; non-causal evidence on its own.")
    md_lines.append("- Freezing from initialization is a different test from freezing after memorization.")
    md_lines.append("")
    md_lines.append("## 11. Recommended next steps")
    md_lines.append("")
    md_lines.append("- Freeze M0 *after* memorization rather than from init.")
    md_lines.append("- Freeze M1 or M2 to test compensation directionality.")
    md_lines.append("- Run ablation on the trained frozen models to test causal sufficiency.")
    md_lines.append("- If 3 seeds is too few for any conclusion, extend to seeds 3-5.")

    with open(os.path.join(out_root, "layer0_freeze_modadd_results_for_chatgpt.md"), "w") as f:
        f.write("\n".join(md_lines))

    # ----- JSON companion -----
    json_runs = []
    for r in rows:
        json_runs.append({
            "run_name": f"{r['variant']}_seed{r['seed']}",
            "variant": r["variant"], "seed": r["seed"],
            "freeze_A0": bool(r["freeze_A0"]), "freeze_M0": bool(r["freeze_M0"]),
            "best_val_acc": None if r["best_val_acc"] in ("","NA",None) else float(r["best_val_acc"]),
            "test_acc_at_best_val":
                None if r["test_acc_at_best_val"] in ("","NA",None) else float(r["test_acc_at_best_val"]),
            "tau_0.90": safe_int(r["tau_0.90"]),
            "tau_0.95": safe_int(r["tau_0.95"]),
            "status": r["status"],
        })
    out_json = {
        "num_runs": 12,
        "seeds": SEEDS,
        "gpu_ids": [1,2,3,4,5,6,7],
        "runs": json_runs,
        "aggregate_by_variant": agg_rows,
        "main_findings": [],
        "recommended_next_steps": [
            "Freeze M0 after memorization rather than from init",
            "Test M1/M2 freezes to localize compensation",
            "Extend to additional seeds if 3 is insufficient",
        ],
    }
    with open(os.path.join(out_root, "layer0_freeze_modadd_results_for_chatgpt.json"), "w") as f:
        json.dump(out_json, f, indent=2, default=float)

    print(f"Created {os.path.join(out_root, 'layer0_freeze_modadd_results_for_chatgpt.md')}")
    print(f"Created {os.path.join(out_root, 'layer0_freeze_modadd_results_for_chatgpt.json')}")
    print(f"plots -> {plot_dir}")
    print("Upload the markdown report to ChatGPT for interpretation.")


if __name__ == "__main__":
    main()
