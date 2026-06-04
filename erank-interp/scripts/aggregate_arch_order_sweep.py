"""
scripts/aggregate_arch_order_sweep.py

Aggregator + report builder for the balanced A/M architecture-order sweep
(claude_code_arch_order_sweep_6gpu_with_results_report.md).

Reads every per-run directory under outputs/arch_order_sweep_6gpu/runs/, plus
the schedule manifest, and produces:
    architecture_order_summary.csv
    plots/accuracy_vs_a_before_m_score.png
    plots/grokking_status_by_schedule_score.png
    plots/accuracy_vs_num_alternations.png
    plots/depth{2,3,4}_componentwise_erank_heatmap.png
    plots/endpoint_sum_delta_A_M_by_schedule.png
    plots/representative_training_time_erank_curves.png
    plots/same_sum_ratio_over_time_representative.png
    plots/erank_vs_top5_mass_representative.png
    arch_order_sweep_results_for_chatgpt.md
    arch_order_sweep_results_for_chatgpt.json

Works on partial sweeps too (writes REPORT_STATUS: INTERIM if any run is still
RUNNING or any manifest schedule is missing a run dir).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from collections import defaultdict
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DONE_STATUSES = {"COMPLETED", "GROKKED", "MEMORIZED_ONLY", "FAILED_TO_FIT", "PARTIAL"}


def safe_float(x: Any) -> float:
    if x is None or x == "" or (isinstance(x, str) and x.strip() == ""):
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def safe_int(x: Any) -> int | None:
    if x is None or x == "":
        return None
    try:
        return int(x)
    except Exception:
        return None


def load_manifest(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_run_dir(run_dir: str) -> dict | None:
    """Load whatever artifacts exist for a single run; tolerate missing files."""
    if not os.path.isdir(run_dir):
        return None
    out: dict = {"run_dir": run_dir}
    status_path = os.path.join(run_dir, "status.txt")
    if os.path.exists(status_path):
        with open(status_path) as f:
            out["status_file"] = f.read().strip().upper()
    else:
        out["status_file"] = "MISSING"

    summary_path = os.path.join(run_dir, "run_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            out["summary"] = json.load(f)

    # Componentwise eRank (per-block deltas).
    cw_path = os.path.join(run_dir, "componentwise_erank.csv")
    if os.path.exists(cw_path):
        with open(cw_path) as f:
            out["componentwise"] = list(csv.DictReader(f))

    # Same-sum clustering.
    ss_path = os.path.join(run_dir, "same_sum_clustering.csv")
    if os.path.exists(ss_path):
        with open(ss_path) as f:
            out["same_sum"] = list(csv.DictReader(f))

    # Spectrum metrics.
    sp_path = os.path.join(run_dir, "spectrum_metrics.csv")
    if os.path.exists(sp_path):
        with open(sp_path) as f:
            out["spectrum"] = list(csv.DictReader(f))

    # Training-time eRank.
    tt_path = os.path.join(run_dir, "training_time_erank.csv")
    if os.path.exists(tt_path):
        with open(tt_path) as f:
            out["training_time_erank"] = list(csv.DictReader(f))

    # Training log (for representative curves).
    tl_path = os.path.join(run_dir, "training_log.csv")
    if os.path.exists(tl_path):
        with open(tl_path) as f:
            out["training_log"] = list(csv.DictReader(f))
    return out


def build_summary_row(manifest_row: dict, run_data: dict | None) -> dict:
    """One row per schedule in the manifest. Missing runs -> mostly NaN."""
    row = dict(manifest_row)
    sched = manifest_row["schedule"]
    nA = int(manifest_row["num_attention"])
    nM = int(manifest_row["num_mlp"])

    # Defaults.
    row["best_val_epoch"] = ""
    row["best_val_acc"] = float("nan")
    row["test_acc_at_best_val"] = float("nan")
    row["final_train_acc"] = float("nan")
    row["final_val_acc"] = float("nan")
    row["final_epoch"] = ""
    row["grokking_tau"] = ""
    row["status"] = "PENDING"
    row["param_count"] = ""
    row["final_rep_erank_best_val"] = float("nan")
    row["final_rep_erank_final"] = float("nan")
    row["sum_delta_A_best_val"] = float("nan")
    row["sum_delta_M_best_val"] = float("nan")
    row["mean_delta_A_best_val"] = float("nan")
    row["mean_delta_M_best_val"] = float("nan")
    row["same_sum_ratio_best_val"] = float("nan")
    row["top5_mass_best_val"] = float("nan")
    row["top10_mass_best_val"] = float("nan")

    if run_data is None:
        return row

    s = run_data.get("summary")
    if s:
        row["best_val_epoch"] = s.get("best_val_epoch") if s.get("best_val_epoch") is not None else ""
        row["best_val_acc"] = safe_float(s.get("best_val_acc"))
        row["test_acc_at_best_val"] = safe_float(s.get("test_acc_at_best_val"))
        row["final_train_acc"] = safe_float(s.get("final_train_acc"))
        row["final_val_acc"] = safe_float(s.get("final_val_acc"))
        row["final_epoch"] = s.get("final_epoch") if s.get("final_epoch") is not None else ""
        row["grokking_tau"] = s.get("grokking_tau") if s.get("grokking_tau") is not None else ""
        row["status"] = s.get("status", run_data.get("status_file", "UNKNOWN"))
        row["param_count"] = s.get("n_params") if s.get("n_params") is not None else ""
    else:
        row["status"] = run_data.get("status_file", "UNKNOWN")

    # Componentwise eRank @ best-val -> sum/mean delta_A/M and final_rep_erank.
    cw = run_data.get("componentwise") or []
    bv_rows = [r for r in cw if r["checkpoint_label"] == "best_val"]
    f_rows = [r for r in cw if r["checkpoint_label"] == "final"]

    def _agg_block_rows(rows: list[dict]) -> dict:
        if not rows:
            return {}
        rows_sorted = sorted(rows, key=lambda r: int(r["block_index"]))
        a_deltas = [safe_float(r["delta_erank"]) for r in rows_sorted if r["block_type"] == "A"]
        m_deltas = [safe_float(r["delta_erank"]) for r in rows_sorted if r["block_type"] == "M"]
        last = rows_sorted[-1]
        return {
            "sum_delta_A": float(np.nansum(a_deltas)) if a_deltas else float("nan"),
            "sum_delta_M": float(np.nansum(m_deltas)) if m_deltas else float("nan"),
            "mean_delta_A": float(np.nanmean(a_deltas)) if a_deltas else float("nan"),
            "mean_delta_M": float(np.nanmean(m_deltas)) if m_deltas else float("nan"),
            "final_rep_erank": safe_float(last["erank_after"]),
        }

    bv_agg = _agg_block_rows(bv_rows)
    f_agg = _agg_block_rows(f_rows)
    row["sum_delta_A_best_val"] = bv_agg.get("sum_delta_A", float("nan"))
    row["sum_delta_M_best_val"] = bv_agg.get("sum_delta_M", float("nan"))
    row["mean_delta_A_best_val"] = bv_agg.get("mean_delta_A", float("nan"))
    row["mean_delta_M_best_val"] = bv_agg.get("mean_delta_M", float("nan"))
    row["final_rep_erank_best_val"] = bv_agg.get("final_rep_erank", float("nan"))
    row["final_rep_erank_final"] = f_agg.get("final_rep_erank", float("nan"))

    # Same-sum ratio @ best-val.
    ss = run_data.get("same_sum") or []
    bv_ss = [r for r in ss if r["checkpoint_label"] == "best_val"]
    if bv_ss:
        row["same_sum_ratio_best_val"] = safe_float(bv_ss[0].get("within_between_ratio"))

    # Spectrum @ best-val.
    sp = run_data.get("spectrum") or []
    bv_sp = [r for r in sp if r["checkpoint_label"] == "best_val"]
    if bv_sp:
        row["top5_mass_best_val"] = safe_float(bv_sp[0].get("top5_mass"))
        row["top10_mass_best_val"] = safe_float(bv_sp[0].get("top10_mass"))

    return row


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

STATUS_COLORS = {
    "GROKKED": "tab:green",
    "PARTIAL": "tab:olive",
    "MEMORIZED_ONLY": "tab:orange",
    "FAILED_TO_FIT": "tab:red",
    "PENDING": "tab:gray",
    "RUNNING": "tab:cyan",
    "FAILED_ERROR": "black",
    "UNKNOWN": "tab:gray",
    "MISSING": "tab:gray",
}

STATUS_MARKERS = {
    "GROKKED": "o",
    "PARTIAL": "P",
    "MEMORIZED_ONLY": "s",
    "FAILED_TO_FIT": "X",
    "PENDING": ".",
    "RUNNING": ".",
    "FAILED_ERROR": "x",
    "UNKNOWN": ".",
    "MISSING": ".",
}


def plot_accuracy_vs_score(summary_rows: list[dict], plot_dir: str) -> list[str]:
    """Scatter test_acc vs a_before_m_score, coloured by depth, marker by status."""
    paths: list[str] = []
    fig, ax = plt.subplots(figsize=(7.5, 5))
    depths = sorted({int(r["depth"]) for r in summary_rows})
    depth_cmap = {2: "tab:blue", 3: "tab:purple", 4: "tab:brown"}
    for r in summary_rows:
        score = safe_float(r["a_before_m_score"])
        acc = safe_float(r["test_acc_at_best_val"])
        if np.isnan(acc):
            continue
        d = int(r["depth"])
        st = r["status"]
        ax.scatter(score, acc, c=depth_cmap.get(d, "tab:gray"),
                   marker=STATUS_MARKERS.get(st, "o"),
                   s=55, edgecolor="black", linewidth=0.4,
                   alpha=0.85, label=None)
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=f"depth {d}",
                          markerfacecolor=depth_cmap.get(d, "tab:gray"),
                          markersize=8, markeredgecolor="black", markeredgewidth=0.4)
               for d in depths]
    for st in ["GROKKED", "PARTIAL", "MEMORIZED_ONLY", "FAILED_TO_FIT"]:
        handles.append(plt.Line2D([0], [0], marker=STATUS_MARKERS[st], color="black",
                                  label=st, markerfacecolor="white",
                                  markersize=8, linestyle=""))
    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.set_xlabel("a_before_m_score  (1.0 = all A before all M)")
    ax.set_ylabel("test_acc @ best-val")
    ax.set_title("Accuracy vs routing-before-processing score")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(plot_dir, "accuracy_vs_a_before_m_score.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # Second view: status counts vs a_before_m_score bucket.
    bins = np.linspace(0.0, 1.0, 6)
    counts: dict[str, list[int]] = defaultdict(lambda: [0] * (len(bins) - 1))
    for r in summary_rows:
        score = safe_float(r["a_before_m_score"])
        if np.isnan(score):
            continue
        bi = min(np.searchsorted(bins, score, side="right") - 1, len(bins) - 2)
        bi = max(bi, 0)
        counts[r["status"]][bi] += 1
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
    bottom = np.zeros(len(bins) - 1)
    for st in ["GROKKED", "PARTIAL", "MEMORIZED_ONLY", "FAILED_TO_FIT",
               "FAILED_ERROR", "PENDING"]:
        if st not in counts:
            continue
        vals = np.array(counts[st])
        ax2.bar((bins[:-1] + bins[1:]) / 2, vals, width=0.18,
                bottom=bottom, label=st, color=STATUS_COLORS.get(st, "tab:gray"),
                edgecolor="black", linewidth=0.4)
        bottom += vals
    ax2.set_xlabel("a_before_m_score bucket")
    ax2.set_ylabel("# runs")
    ax2.set_title("Grokking status by routing-before-processing score")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    p2 = os.path.join(plot_dir, "grokking_status_by_schedule_score.png")
    fig2.savefig(p2, dpi=140)
    plt.close(fig2)
    paths.append(p2)
    return paths


def plot_accuracy_vs_alternations(summary_rows: list[dict], plot_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    depth_cmap = {2: "tab:blue", 3: "tab:purple", 4: "tab:brown"}
    for r in summary_rows:
        n_alt = safe_int(r["num_alternations"])
        acc = safe_float(r["test_acc_at_best_val"])
        if np.isnan(acc) or n_alt is None:
            continue
        d = int(r["depth"])
        # jitter for legibility
        jitter = (np.random.rand() - 0.5) * 0.25
        ax.scatter(n_alt + jitter, acc, c=depth_cmap.get(d, "tab:gray"),
                   marker=STATUS_MARKERS.get(r["status"], "o"),
                   s=55, edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.set_xlabel("num_alternations (jittered)")
    ax.set_ylabel("test_acc @ best-val")
    ax.set_title("Accuracy vs alternations between A and M blocks")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(plot_dir, "accuracy_vs_num_alternations.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_componentwise_heatmaps(
    summary_rows: list[dict],
    run_data_map: dict[str, dict],
    plot_dir: str,
) -> list[str]:
    """One heatmap per depth: rows=schedules sorted by a_before_m_score, cols=block index."""
    paths: list[str] = []
    by_depth: dict[int, list[dict]] = defaultdict(list)
    for r in summary_rows:
        by_depth[int(r["depth"])].append(r)
    for d, rows in sorted(by_depth.items()):
        rows = sorted(rows, key=lambda r: (-safe_float(r["a_before_m_score"]),
                                            r["schedule"]))
        n_blocks = 2 * d
        mat = np.full((len(rows), n_blocks), np.nan, dtype=np.float32)
        labels: list[str] = []
        kinds: list[list[str]] = []
        for ri, r in enumerate(rows):
            labels.append(f"{r['schedule']} (s={safe_float(r['a_before_m_score']):.2f})")
            run = run_data_map.get(r["run_id"])
            if not run or not run.get("componentwise"):
                kinds.append(["?"] * n_blocks)
                continue
            cw = [c for c in run["componentwise"] if c["checkpoint_label"] == "best_val"]
            cw_sorted = sorted(cw, key=lambda c: int(c["block_index"]))
            kinds_row: list[str] = ["?"] * n_blocks
            for c in cw_sorted:
                bi = int(c["block_index"])
                mat[ri, bi] = safe_float(c["delta_erank"])
                kinds_row[bi] = c["block_type"]
            kinds.append(kinds_row)

        # Plot.
        fig_h = max(4.5, 0.22 * len(rows) + 1.2)
        fig, ax = plt.subplots(figsize=(max(6, 0.9 * n_blocks + 4), fig_h))
        vmax = float(np.nanmax(np.abs(mat))) if np.any(~np.isnan(mat)) else 1.0
        if vmax == 0 or np.isnan(vmax):
            vmax = 1.0
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xticks(np.arange(n_blocks))
        ax.set_xticklabels([f"b{i}" for i in range(n_blocks)], fontsize=8)
        ax.set_xlabel("block index")
        ax.set_title(f"depth {d}: componentwise Δ_eRank @ best-val")
        # annotate kinds on each cell.
        for ri in range(len(rows)):
            for bi in range(n_blocks):
                kind = kinds[ri][bi] if ri < len(kinds) else "?"
                v = mat[ri, bi]
                txt = f"{kind}\n{v:+.1f}" if not np.isnan(v) else kind
                ax.text(bi, ri, txt, ha="center", va="center", fontsize=5.5,
                        color="black")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Δ_eRank")
        fig.tight_layout()
        p = os.path.join(plot_dir, f"depth{d}_componentwise_erank_heatmap.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_endpoint_sum_delta_AM(summary_rows: list[dict], plot_dir: str) -> str:
    rows = [r for r in summary_rows
            if not np.isnan(safe_float(r["sum_delta_A_best_val"]))
            and not np.isnan(safe_float(r["sum_delta_M_best_val"]))]
    rows = sorted(rows, key=lambda r: (int(r["depth"]), r["schedule"]))
    if not rows:
        # write a placeholder so the path exists.
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "no completed runs yet", ha="center", va="center")
        p = os.path.join(plot_dir, "endpoint_sum_delta_A_M_by_schedule.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        return p
    sched = [r["schedule"] for r in rows]
    sda = [safe_float(r["sum_delta_A_best_val"]) for r in rows]
    sdm = [safe_float(r["sum_delta_M_best_val"]) for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(7, 0.18 * len(rows) + 3), 4))
    w = 0.4
    ax.bar(x - w / 2, sda, width=w, label="Σ Δ_A", color="tab:blue", edgecolor="black", linewidth=0.3)
    ax.bar(x + w / 2, sdm, width=w, label="Σ Δ_M", color="tab:orange", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(sched, rotation=90, fontsize=6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Σ Δ_eRank")
    ax.set_title("Endpoint Σ Δ_A vs Σ Δ_M per schedule @ best-val")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(plot_dir, "endpoint_sum_delta_A_M_by_schedule.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def _pick_representative_runs(summary_rows: list[dict]) -> list[dict]:
    """Pick a small set of representative runs for time-series plots."""
    picks: list[dict] = []
    by_depth: dict[int, list[dict]] = defaultdict(list)
    for r in summary_rows:
        by_depth[int(r["depth"])].append(r)
    seen_ids: set[str] = set()
    for d, rows in sorted(by_depth.items()):
        # Standard interleaved (AM...AM)
        for r in rows:
            n_alt = safe_int(r["num_alternations"]) or 0
            score = safe_float(r["a_before_m_score"])
            if r["schedule"] in (
                "A" + "M" * (2 * d - 1),  # unusual
                "AMAM"[: 2 * d],          # depth-2 only
                "AMAMAMAM"[: 2 * d],      # generic interleaved
            ):
                if r["run_id"] not in seen_ids:
                    picks.append(r); seen_ids.add(r["run_id"])
                    break
        # Highest a_before_m_score (i.e. AAAA...MMMM)
        rows_sorted = sorted(
            rows, key=lambda r: (-safe_float(r["a_before_m_score"]), r["schedule"])
        )
        for r in rows_sorted[:1]:
            if r["run_id"] not in seen_ids:
                picks.append(r); seen_ids.add(r["run_id"])
        # Lowest a_before_m_score (MMMM...AAAA)
        rows_sorted_low = sorted(
            rows, key=lambda r: (safe_float(r["a_before_m_score"]), r["schedule"])
        )
        for r in rows_sorted_low[:1]:
            if r["run_id"] not in seen_ids:
                picks.append(r); seen_ids.add(r["run_id"])
        # All-A control + all-M control.
        for r in rows:
            if r.get("is_control") == "1" and r["run_id"] not in seen_ids:
                picks.append(r); seen_ids.add(r["run_id"])
    return picks


def plot_representative_training_time_erank(
    summary_rows: list[dict],
    run_data_map: dict[str, dict],
    plot_dir: str,
) -> str:
    picks = _pick_representative_runs(summary_rows)
    fig, (ax_a, ax_e) = plt.subplots(1, 2, figsize=(11, 4.5))
    for r in picks:
        run = run_data_map.get(r["run_id"])
        if not run:
            continue
        log_rows = run.get("training_log") or []
        if not log_rows:
            continue
        ep = [safe_float(x["epoch"]) for x in log_rows]
        va = [safe_float(x["val_acc"]) for x in log_rows]
        ax_a.plot(ep, va, label=r["schedule"], alpha=0.85, linewidth=1.2)
        tt = run.get("training_time_erank") or []
        if tt:
            ep_e = [safe_float(x["epoch"]) for x in tt]
            er = [safe_float(x["final_rep_erank"]) for x in tt]
            ax_e.plot(ep_e, er, label=r["schedule"], alpha=0.85, linewidth=1.2)
    ax_a.set_xlabel("epoch")
    ax_a.set_ylabel("val_acc")
    ax_a.set_title("Val accuracy — representative schedules")
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.legend(fontsize=7, ncol=2)
    ax_a.grid(True, alpha=0.3)
    ax_e.set_xlabel("epoch")
    ax_e.set_ylabel("final_rep_eRank")
    ax_e.set_title("Final-rep eRank over training")
    ax_e.legend(fontsize=7, ncol=2)
    ax_e.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(plot_dir, "representative_training_time_erank_curves.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_same_sum_ratio_over_time(
    summary_rows: list[dict],
    run_data_map: dict[str, dict],
    plot_dir: str,
) -> str:
    picks = _pick_representative_runs(summary_rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in picks:
        run = run_data_map.get(r["run_id"])
        if not run:
            continue
        ss = run.get("same_sum") or []
        if not ss:
            continue
        # Group by epoch order.
        ss_sorted = sorted(ss, key=lambda c: safe_int(c["epoch"]) or 0)
        epochs = [safe_int(c["epoch"]) for c in ss_sorted]
        ratios = [safe_float(c["within_between_ratio"]) for c in ss_sorted]
        ax.plot(epochs, ratios, marker="o", label=r["schedule"], alpha=0.85)
    ax.set_xlabel("epoch")
    ax.set_ylabel("D_within / D_between")
    ax.set_title("Same-sum clustering ratio — representative schedules")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(plot_dir, "same_sum_ratio_over_time_representative.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_erank_vs_top5_mass(
    summary_rows: list[dict], plot_dir: str,
) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    depth_cmap = {2: "tab:blue", 3: "tab:purple", 4: "tab:brown"}
    for r in summary_rows:
        e = safe_float(r["final_rep_erank_best_val"])
        m5 = safe_float(r["top5_mass_best_val"])
        if np.isnan(e) or np.isnan(m5):
            continue
        d = int(r["depth"])
        ax.scatter(e, m5, c=depth_cmap.get(d, "tab:gray"),
                   marker=STATUS_MARKERS.get(r["status"], "o"),
                   s=55, edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.set_xlabel("final_rep_eRank @ best-val")
    ax.set_ylabel("top-5 singular mass")
    ax.set_title("eRank vs top-5 mass @ best-val")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(plot_dir, "erank_vs_top5_mass_representative.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_summary_csv(summary_rows: list[dict], out_path: str) -> None:
    # Stable column order.
    fieldnames = [
        "run_id", "depth", "schedule", "seed_implicit",
        "num_blocks", "num_attention", "num_mlp",
        "num_alternations", "a_before_m_score",
        "first_block", "last_block", "param_count",
        "best_val_epoch", "best_val_acc", "test_acc_at_best_val",
        "final_train_acc", "final_val_acc", "final_epoch",
        "grokking_tau", "status",
        "final_rep_erank_best_val", "final_rep_erank_final",
        "sum_delta_A_best_val", "sum_delta_M_best_val",
        "mean_delta_A_best_val", "mean_delta_M_best_val",
        "same_sum_ratio_best_val",
        "top5_mass_best_val", "top10_mass_best_val",
        "is_balanced", "is_control",
    ]
    # Add seed_implicit fixed to 0.
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in summary_rows:
            row = dict(r)
            row["seed_implicit"] = 0
            w.writerow(row)


def _pearson(xs: list[float], ys: list[float]) -> float:
    xs_a = np.array(xs, dtype=float)
    ys_a = np.array(ys, dtype=float)
    mask = ~np.isnan(xs_a) & ~np.isnan(ys_a)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(xs_a[mask], ys_a[mask])[0, 1])


def _spearman(xs: list[float], ys: list[float]) -> float:
    xs_a = np.array(xs, dtype=float)
    ys_a = np.array(ys, dtype=float)
    mask = ~np.isnan(xs_a) & ~np.isnan(ys_a)
    if mask.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(xs_a[mask]))
    ry = np.argsort(np.argsort(ys_a[mask]))
    return float(np.corrcoef(rx, ry)[0, 1])


def _fmt(x: Any, prec: int = 4) -> str:
    if x is None or x == "":
        return "NA"
    if isinstance(x, float) and np.isnan(x):
        return "NA"
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)


def build_markdown_report(
    summary_rows: list[dict],
    manifest_rows: list[dict],
    plot_paths: dict[str, str],
    out_root: str,
    gpu_info: str,
) -> str:
    n_total = len(manifest_rows)
    statuses = [r["status"] for r in summary_rows]
    n_pending = sum(1 for s in statuses if s in ("PENDING", "RUNNING"))
    n_failed = sum(1 for s in statuses if s == "FAILED_ERROR")
    n_done = sum(1 for s in statuses if s in DONE_STATUSES)
    n_grokked = sum(1 for s in statuses if s == "GROKKED")
    report_status = "FINAL" if n_pending == 0 else "INTERIM"

    # Status counts by depth.
    status_by_depth: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in summary_rows:
        status_by_depth[int(r["depth"])][r["status"]] += 1

    # Sort table by depth, status (GROKKED first), test_acc desc.
    def _sort_key(r: dict):
        s_rank = {"GROKKED": 0, "PARTIAL": 1, "MEMORIZED_ONLY": 2,
                  "FAILED_TO_FIT": 3, "FAILED_ERROR": 4, "PENDING": 5,
                  "RUNNING": 5, "UNKNOWN": 6, "MISSING": 6}.get(r["status"], 7)
        acc = safe_float(r["test_acc_at_best_val"])
        return (int(r["depth"]), s_rank, -(acc if not np.isnan(acc) else -1))
    sorted_rows = sorted(summary_rows, key=_sort_key)

    # Correlations across runs.
    scores = [safe_float(r["a_before_m_score"]) for r in summary_rows]
    accs = [safe_float(r["test_acc_at_best_val"]) for r in summary_rows]
    n_alts = [safe_float(r["num_alternations"]) for r in summary_rows]
    eranks = [safe_float(r["final_rep_erank_best_val"]) for r in summary_rows]
    ss_ratios = [safe_float(r["same_sum_ratio_best_val"]) for r in summary_rows]
    taus = [safe_float(r["grokking_tau"]) if r["status"] == "GROKKED"
            else float("nan") for r in summary_rows]

    corr_score_acc_p = _pearson(scores, accs)
    corr_score_acc_s = _spearman(scores, accs)
    corr_alt_acc_p = _pearson(n_alts, accs)
    corr_alt_acc_s = _spearman(n_alts, accs)
    corr_score_tau_p = _pearson(scores, taus)
    corr_erank_acc_p = _pearson(eranks, accs)
    corr_ss_acc_p = _pearson(ss_ratios, accs)

    # Build markdown.
    lines: list[str] = []
    lines.append("# Architecture-Order Sweep Results for ChatGPT")
    lines.append("")
    lines.append(f"REPORT_STATUS: **{report_status}**")
    lines.append("")
    lines.append("## 1. Report status")
    lines.append("")
    lines.append(f"- timestamp: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- runs completed (terminal status): {n_done}")
    lines.append(f"- runs still pending/running: {n_pending}")
    lines.append(f"- runs failed (FAILED_ERROR): {n_failed}")
    lines.append(f"- runs grokked: {n_grokked}")
    lines.append(f"- tmux session: arch_order_sweep_6gpu")
    lines.append(f"- GPUs used: see `gpu_assignment.txt` ({gpu_info})")
    lines.append("")
    lines.append("## 2. Executive summary")
    lines.append("")
    top_score = next((r for r in sorted_rows if r["status"] == "GROKKED"), None)
    top_failed_score = None
    for r in sorted_rows:
        score = safe_float(r["a_before_m_score"])
        if r["status"] in ("FAILED_TO_FIT", "MEMORIZED_ONLY") and not np.isnan(score):
            top_failed_score = r
            break
    bullets: list[str] = []
    bullets.append(f"Single-seed sweep over {n_total} schedules ({len(summary_rows)} loaded).")
    bullets.append(f"Best grokked schedule so far: "
                   f"{top_score['schedule'] if top_score else 'none yet'} "
                   f"(test={_fmt(top_score['test_acc_at_best_val']) if top_score else 'NA'}, "
                   f"a_before_m_score="
                   f"{_fmt(top_score['a_before_m_score'], 2) if top_score else 'NA'}).")
    bullets.append(f"Pearson(a_before_m_score, test_acc) = {_fmt(corr_score_acc_p, 3)};  "
                   f"Spearman = {_fmt(corr_score_acc_s, 3)}.")
    bullets.append(f"Pearson(num_alternations, test_acc) = {_fmt(corr_alt_acc_p, 3)};  "
                   f"Spearman = {_fmt(corr_alt_acc_s, 3)}.")
    bullets.append(f"Pearson(final_rep_eRank @ best-val, test_acc) = {_fmt(corr_erank_acc_p, 3)}.")
    bullets.append(f"Pearson(same-sum ratio @ best-val, test_acc) = {_fmt(corr_ss_acc_p, 3)}  "
                   "(negative = compression aligns with success).")
    if top_failed_score is not None:
        bullets.append(f"Notable non-grokking: {top_failed_score['schedule']} "
                       f"(status={top_failed_score['status']}, score="
                       f"{_fmt(top_failed_score['a_before_m_score'], 2)}).")
    bullets.append("Caveats: single seed; architectures NOT parameter-matched.")
    for b in bullets:
        lines.append(f"- {b}")
    lines.append("")

    lines.append("## 3. Experimental design")
    lines.append("")
    lines.append("- task: modular addition c = a + b mod 113")
    lines.append("- d_model=128, n_heads=4, d_head=32, d_mlp=512, AdamW lr=3e-4, wd=1.0, batch=256, seed=0")
    lines.append("- balanced permutations of A and M blocks: depth 2 (6), depth 3 (20), depth 4 (70)")
    lines.append("- optional controls: all-A and all-M for each depth (6 total)")
    lines.append("- validation-based checkpoint selection; test only at best-val")
    lines.append("- early stop at val_acc >= 0.95 sustained for 200 evals")
    lines.append("")

    lines.append("## 4. Schedule manifest summary")
    lines.append("")
    lines.append("| depth | num schedules | completed | grokked | memorized | failed_to_fit | partial | failed_error | pending |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in sorted(status_by_depth.keys()):
        sd = status_by_depth[d]
        completed = sum(sd.get(k, 0) for k in DONE_STATUSES)
        lines.append(
            f"| {d} | "
            f"{sum(1 for r in summary_rows if int(r['depth']) == d)} | "
            f"{completed} | "
            f"{sd.get('GROKKED', 0)} | "
            f"{sd.get('MEMORIZED_ONLY', 0)} | "
            f"{sd.get('FAILED_TO_FIT', 0)} | "
            f"{sd.get('PARTIAL', 0)} | "
            f"{sd.get('FAILED_ERROR', 0)} | "
            f"{sd.get('PENDING', 0) + sd.get('RUNNING', 0)} |"
        )
    lines.append("")

    lines.append("## 5. Accuracy and grokking summary")
    lines.append("")
    lines.append("| run_id | depth | schedule | a_before_m | n_alt | best_val | test@best_val | final_train | final_val | tau | status |")
    lines.append("|:--|---:|:--|---:|---:|---:|---:|---:|---:|---:|:--|")
    for r in sorted_rows:
        lines.append(
            f"| {r['run_id']} | {r['depth']} | {r['schedule']} | "
            f"{_fmt(safe_float(r['a_before_m_score']), 2)} | "
            f"{r['num_alternations']} | "
            f"{_fmt(safe_float(r['best_val_acc']))} | "
            f"{_fmt(safe_float(r['test_acc_at_best_val']))} | "
            f"{_fmt(safe_float(r['final_train_acc']))} | "
            f"{_fmt(safe_float(r['final_val_acc']))} | "
            f"{r['grokking_tau']} | {r['status']} |"
        )
    lines.append("")

    lines.append("## 6. Which schedules solved?")
    lines.append("")
    for st in ("GROKKED", "PARTIAL", "MEMORIZED_ONLY", "FAILED_TO_FIT",
               "FAILED_ERROR", "PENDING"):
        rows_st = [r for r in sorted_rows if r["status"] == st]
        if not rows_st:
            continue
        lines.append(f"### {st} ({len(rows_st)})")
        for r in rows_st:
            lines.append(
                f"- `{r['schedule']}` (depth {r['depth']}, score "
                f"{_fmt(safe_float(r['a_before_m_score']), 2)}, alt "
                f"{r['num_alternations']}, "
                f"test {_fmt(safe_float(r['test_acc_at_best_val']))})"
            )
        lines.append("")

    lines.append("## 7. Routing-before-processing analysis")
    lines.append("")
    lines.append(f"- Pearson(a_before_m_score, test_acc@best_val) = {_fmt(corr_score_acc_p, 3)}")
    lines.append(f"- Spearman(a_before_m_score, test_acc@best_val) = {_fmt(corr_score_acc_s, 3)}")
    lines.append(f"- Pearson(a_before_m_score, grokking_tau) (grokked only) = {_fmt(corr_score_tau_p, 3)}")
    lines.append("")
    if not np.isnan(corr_score_acc_p):
        if corr_score_acc_p > 0.3:
            lines.append("Interpretation: high A-before-M score is positively associated with "
                         "grokking in this single-seed sweep. Phrase carefully as suggestive.")
        elif corr_score_acc_p < -0.3:
            lines.append("Interpretation: high A-before-M score is *negatively* associated with "
                         "grokking — the data suggest processing-before-routing or interleaving helps.")
        else:
            lines.append("Interpretation: routing-before-processing does not clearly predict "
                         "grokking; alternation or capacity may be the dominant factor.")
        lines.append("")

    lines.append("## 8. Interleaving analysis")
    lines.append("")
    lines.append(f"- Pearson(num_alternations, test_acc@best_val) = {_fmt(corr_alt_acc_p, 3)}")
    lines.append(f"- Spearman(num_alternations, test_acc@best_val) = {_fmt(corr_alt_acc_s, 3)}")
    lines.append("")

    lines.append("## 9. Componentwise eRank results")
    lines.append("")
    lines.append("| run_id | depth | schedule | block | type | erank_before | erank_after | Δ_erank |")
    lines.append("|:--|---:|:--|---:|:--|---:|---:|---:|")
    # Choose representative subset for table size: best grokked, best failed, etc.
    chosen: list[dict] = []
    if top_score is not None:
        chosen.append(top_score)
    if top_failed_score is not None:
        chosen.append(top_failed_score)
    # Add depth-{2,3,4} highest-score and lowest-score grokked or run.
    for d in sorted(status_by_depth.keys()):
        depth_rows = [r for r in summary_rows if int(r["depth"]) == d]
        if depth_rows:
            top = max(depth_rows, key=lambda r: safe_float(r["a_before_m_score"]) if not np.isnan(safe_float(r["a_before_m_score"])) else -1)
            low = min(depth_rows, key=lambda r: safe_float(r["a_before_m_score"]) if not np.isnan(safe_float(r["a_before_m_score"])) else 999)
            for r in (top, low):
                if r not in chosen:
                    chosen.append(r)

    # Deduplicate by run_id.
    seen = set(); chosen_dedup: list[dict] = []
    for r in chosen:
        if r["run_id"] not in seen:
            seen.add(r["run_id"])
            chosen_dedup.append(r)

    # Need run_data_map for per-block rows; load once.
    # We'll pass via closure in caller; here we re-load the componentwise CSVs ourselves.
    runs_dir = os.path.join(out_root, "runs")
    for r in chosen_dedup:
        cw_path = os.path.join(runs_dir, r["run_id"], "componentwise_erank.csv")
        if not os.path.exists(cw_path):
            continue
        with open(cw_path) as f:
            rows_cw = [row for row in csv.DictReader(f)
                       if row["checkpoint_label"] == "best_val"]
        for row in sorted(rows_cw, key=lambda x: int(x["block_index"])):
            lines.append(
                f"| {r['run_id']} | {r['depth']} | {r['schedule']} | "
                f"{row['block_index']} | {row['block_type']} | "
                f"{_fmt(safe_float(row['erank_before']), 3)} | "
                f"{_fmt(safe_float(row['erank_after']), 3)} | "
                f"{_fmt(safe_float(row['delta_erank']), 3)} |"
            )
    lines.append("")
    lines.append("Aggregate per-schedule:")
    lines.append("")
    lines.append("| run_id | sum_Δ_A | sum_Δ_M | mean_Δ_A | mean_Δ_M | final_rep_eRank |")
    lines.append("|:--|---:|---:|---:|---:|---:|")
    for r in chosen_dedup:
        lines.append(
            f"| {r['run_id']} | "
            f"{_fmt(safe_float(r['sum_delta_A_best_val']), 3)} | "
            f"{_fmt(safe_float(r['sum_delta_M_best_val']), 3)} | "
            f"{_fmt(safe_float(r['mean_delta_A_best_val']), 3)} | "
            f"{_fmt(safe_float(r['mean_delta_M_best_val']), 3)} | "
            f"{_fmt(safe_float(r['final_rep_erank_best_val']), 3)} |"
        )
    lines.append("")

    lines.append("## 10. Training-time eRank and grokking dynamics")
    lines.append("")
    lines.append("See `plots/representative_training_time_erank_curves.png` and per-run "
                 "`training_time_erank.csv`.")
    lines.append("")
    lines.append("## 11. Same-sum clustering and spectral compression")
    lines.append("")
    lines.append("| run_id | ratio @ best-val | top5_mass | top10_mass | final_rep_eRank |")
    lines.append("|:--|---:|---:|---:|---:|")
    for r in chosen_dedup:
        lines.append(
            f"| {r['run_id']} | "
            f"{_fmt(safe_float(r['same_sum_ratio_best_val']), 3)} | "
            f"{_fmt(safe_float(r['top5_mass_best_val']), 3)} | "
            f"{_fmt(safe_float(r['top10_mass_best_val']), 3)} | "
            f"{_fmt(safe_float(r['final_rep_erank_best_val']), 3)} |"
        )
    lines.append("")

    lines.append("## 12. Plot inventory")
    lines.append("")
    lines.append("| plot path | what it shows | ready_for_slides |")
    lines.append("|:--|:--|:-:|")
    for label, p in plot_paths.items():
        if not p:
            continue
        rel = os.path.relpath(p, PROJECT_ROOT)
        lines.append(f"| `{rel}` | {label} | yes |")
    lines.append("")

    lines.append("## 13. Current tmux/GPU status")
    lines.append("")
    lines.append(f"- tmux session: `arch_order_sweep_6gpu`")
    lines.append(f"- {gpu_info}")
    lines.append("- See `outputs/arch_order_sweep_6gpu/logs/tmux_master.log` for the dispatch log.")
    lines.append("")

    lines.append("## 14. Caveats and possible issues")
    lines.append("")
    lines.append("- Single seed (seed=0); no multi-seed reliability check.")
    lines.append("- Architectures are NOT parameter-matched (A-only blocks have no MLP; M-only blocks have no attention).")
    lines.append("- Failed-model eRank is not the geometry of a successful circuit.")
    lines.append("- BlockScheduleTransformer is a custom implementation (no LayerNorm, GELU MLPs, learned positional embeddings, causal mask). Absolute numbers may differ slightly from HookedTransformer baselines.")
    lines.append("- Same-sum clustering uses centroid-based between-class distance.")
    lines.append("")

    lines.append("## 15. Thesis interpretation")
    lines.append("")
    if not np.isnan(corr_score_acc_p):
        if corr_score_acc_p > 0.3:
            lines.append("The single-seed sweep suggests routing-before-processing schedules are "
                         "more likely to grok on modular addition. Re-run with multiple seeds before "
                         "treating this as proven.")
        elif corr_score_acc_p < -0.3:
            lines.append("The single-seed sweep suggests processing-before-routing or interleaved "
                         "schedules outperform pure routing-then-processing. This complicates the "
                         "simple 'attention routes, MLP processes' story.")
        else:
            lines.append("Block ordering does not strongly predict grokking in this sweep; "
                         "the dominant factor appears to be something other than A/M position.")
    else:
        lines.append("Insufficient completed runs to draw an interpretation yet.")
    lines.append("")

    lines.append("## 16. Recommended next steps")
    lines.append("")
    lines.append("- Re-run top-3 grokked and top-3 failed schedules with seeds 1 and 2.")
    lines.append("- Parameter-match key variants (e.g., AAAA vs AAMM with equalised total params).")
    lines.append("- Dense checkpointing around grokking_tau for the most successful schedules.")
    lines.append("- Investigate any FAILED_ERROR runs and resubmit with `--rerun_failed`.")
    lines.append("- If a clear winner emerges, run an ablation that swaps just one block kind at a time.")
    lines.append("")

    return "\n".join(lines)


def build_json_companion(
    summary_rows: list[dict],
    manifest_rows: list[dict],
    gpu_info: str,
) -> dict:
    statuses = [r["status"] for r in summary_rows]
    n_pending = sum(1 for s in statuses if s in ("PENDING", "RUNNING"))
    n_done = sum(1 for s in statuses if s in DONE_STATUSES)
    n_failed = sum(1 for s in statuses if s == "FAILED_ERROR")
    n_grokked = sum(1 for s in statuses if s == "GROKKED")
    report_status = "FINAL" if n_pending == 0 else "INTERIM"

    runs_json: list[dict] = []
    for r in summary_rows:
        runs_json.append({
            "run_id": r["run_id"],
            "depth": int(r["depth"]),
            "schedule": r["schedule"],
            "a_before_m_score": (None if np.isnan(safe_float(r["a_before_m_score"]))
                                  else safe_float(r["a_before_m_score"])),
            "num_alternations": safe_int(r["num_alternations"]),
            "best_val_acc": (None if np.isnan(safe_float(r["best_val_acc"]))
                              else safe_float(r["best_val_acc"])),
            "test_acc_at_best_val": (None if np.isnan(safe_float(r["test_acc_at_best_val"]))
                                      else safe_float(r["test_acc_at_best_val"])),
            "final_train_acc": (None if np.isnan(safe_float(r["final_train_acc"]))
                                 else safe_float(r["final_train_acc"])),
            "final_val_acc": (None if np.isnan(safe_float(r["final_val_acc"]))
                               else safe_float(r["final_val_acc"])),
            "grokking_tau": safe_int(r["grokking_tau"]),
            "status": r["status"],
            "sum_delta_A_best_val": (None if np.isnan(safe_float(r["sum_delta_A_best_val"]))
                                      else safe_float(r["sum_delta_A_best_val"])),
            "sum_delta_M_best_val": (None if np.isnan(safe_float(r["sum_delta_M_best_val"]))
                                      else safe_float(r["sum_delta_M_best_val"])),
            "final_rep_erank_best_val": (None if np.isnan(safe_float(r["final_rep_erank_best_val"]))
                                          else safe_float(r["final_rep_erank_best_val"])),
            "same_sum_ratio_best_val": (None if np.isnan(safe_float(r["same_sum_ratio_best_val"]))
                                         else safe_float(r["same_sum_ratio_best_val"])),
            "top5_mass_best_val": (None if np.isnan(safe_float(r["top5_mass_best_val"]))
                                    else safe_float(r["top5_mass_best_val"])),
        })

    return {
        "report_status": report_status,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "num_runs_total": len(manifest_rows),
        "num_runs_completed": n_done,
        "num_runs_running": n_pending,
        "num_runs_failed": n_failed,
        "num_grokked": n_grokked,
        "gpu_count_used": 6,
        "tmux_session": "arch_order_sweep_6gpu",
        "gpu_info": gpu_info,
        "top_patterns": [],
        "runs": runs_json,
        "recommended_next_steps": [
            "Run seeds 1 and 2 for the top-3 grokked and top-3 failed schedules.",
            "Parameter-match key variants.",
            "Dense checkpointing around grokking_tau for top schedules.",
            "Resubmit any FAILED_ERROR runs with --rerun_failed.",
            "If a clear ordering emerges, ablate one block kind at a time.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="outputs/arch_order_sweep_6gpu")
    args = parser.parse_args()

    out_root = os.path.join(PROJECT_ROOT, args.output_root)
    manifest_path = os.path.join(out_root, "schedule_manifest.csv")
    plot_dir = os.path.join(out_root, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.exists(manifest_path):
        print(f"Missing manifest: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest_rows = load_manifest(manifest_path)
    runs_dir = os.path.join(out_root, "runs")

    run_data_map: dict[str, dict] = {}
    summary_rows: list[dict] = []
    for m in manifest_rows:
        rd = load_run_dir(os.path.join(runs_dir, m["run_id"]))
        run_data_map[m["run_id"]] = rd or {}
        summary_rows.append(build_summary_row(m, rd))

    # Architecture order summary CSV.
    write_summary_csv(summary_rows,
                      os.path.join(out_root, "architecture_order_summary.csv"))

    # Plots.
    plot_paths: dict[str, str] = {}
    p_pair = plot_accuracy_vs_score(summary_rows, plot_dir)
    plot_paths["Accuracy vs a_before_m_score"] = p_pair[0]
    plot_paths["Status by a_before_m_score bucket"] = p_pair[1]
    plot_paths["Accuracy vs num_alternations"] = plot_accuracy_vs_alternations(summary_rows, plot_dir)
    for p in plot_componentwise_heatmaps(summary_rows, run_data_map, plot_dir):
        plot_paths[os.path.basename(p)] = p
    plot_paths["Endpoint Σ Δ_A vs Σ Δ_M per schedule"] = plot_endpoint_sum_delta_AM(summary_rows, plot_dir)
    plot_paths["Representative training-time eRank curves"] = plot_representative_training_time_erank(
        summary_rows, run_data_map, plot_dir,
    )
    plot_paths["Same-sum ratio over time (representative)"] = plot_same_sum_ratio_over_time(
        summary_rows, run_data_map, plot_dir,
    )
    plot_paths["eRank vs top-5 mass @ best-val"] = plot_erank_vs_top5_mass(summary_rows, plot_dir)

    # GPU info from gpu_assignment.txt.
    gpu_info = "unknown"
    gpu_assign_path = os.path.join(out_root, "gpu_assignment.txt")
    if os.path.exists(gpu_assign_path):
        with open(gpu_assign_path) as f:
            gpu_info = " | ".join(line.strip() for line in f.readlines() if line.strip())

    # Markdown + JSON.
    md = build_markdown_report(summary_rows, manifest_rows, plot_paths, out_root, gpu_info)
    md_path = os.path.join(out_root, "arch_order_sweep_results_for_chatgpt.md")
    with open(md_path, "w") as f:
        f.write(md)

    js = build_json_companion(summary_rows, manifest_rows, gpu_info)
    js_path = os.path.join(out_root, "arch_order_sweep_results_for_chatgpt.json")
    with open(js_path, "w") as f:
        json.dump(js, f, indent=2)

    print(f"Created {md_path}")
    print(f"Created {js_path}")
    print(f"architecture_order_summary.csv -> {os.path.join(out_root, 'architecture_order_summary.csv')}")
    print(f"plots -> {plot_dir}")
    print("Upload the markdown file to ChatGPT for interpretation.")


if __name__ == "__main__":
    main()
