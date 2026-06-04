"""
analysis/hybrid_query_diagnostics.py

Phase A diagnostics for hybrid retrieve-then-add models trained on p=23, m=3.

Goal: determine whether a trained hybrid model is using the query keys to
retrieve values, or whether it has fallen into a "fixed-pair shortcut"
(adding two specific context values regardless of the query).

For each available checkpoint, this script:
  1. Reconstructs the model and the deterministic test set.
  2. Runs inference at the EQ position for every test example.
  3. Decodes (v0, v1, v2, q1_idx, q2_idx) from the sequence.
  4. Computes the three candidate fixed-pair sums s01, s02, s12 mod p.
  5. Records whether the prediction matches each candidate.
  6. Runs a counterfactual test: for each context, evaluates all three
     unordered query pairs and measures whether the prediction tracks the
     query.
  7. Compares model accuracy to fixed-pair baselines.

Outputs are written to outputs/hybrid_query_diagnostics/.
"""

from __future__ import annotations

import csv
import glob
import itertools
import json
import os
import sys
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data.hybrid_retrieve_add import generate_hybrid_data_exhaustive
from models.transformer import create_standard_transformer


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

CANDIDATE_GLOBS = [
    "outputs/hybrid_p23",
    "outputs/hybrid_small_p23_3layer",
    "outputs/hybrid_small_p23_4layer",
    "outputs/hybrid_small_p23_m3_standard_2layer_seed*",
    "outputs/hybrid_small_p23_m3_standard_3layer_seed*",
    "outputs/hybrid_small_p23_m3_standard_4layer_seed*",
]


def discover_checkpoints(project_root: str) -> list[dict[str, Any]]:
    """Return per-model metadata for every p=23 m=3 hybrid checkpoint we can find."""
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pattern in CANDIDATE_GLOBS:
        for path in sorted(glob.glob(os.path.join(project_root, pattern))):
            if not os.path.isdir(path):
                continue
            if path in seen:
                continue
            # Skip stale snapshots used as backups.
            if "_stale_" in os.path.basename(path):
                continue
            seen.add(path)

            best = os.path.join(path, "best_val_checkpoint.pt")
            final = os.path.join(path, "final_checkpoint.pt")
            cfg = os.path.join(path, "config_used.yaml")
            split = os.path.join(path, "split_metadata.json")

            ckpt_path = best if os.path.exists(best) else final if os.path.exists(final) else None
            ckpt_kind = "best_val" if ckpt_path == best else "final" if ckpt_path == final else None
            if ckpt_path is None:
                continue
            if not os.path.exists(cfg):
                continue

            with open(cfg, "r") as f:
                cfg_dict = yaml.safe_load(f)

            split_dict = None
            if os.path.exists(split):
                with open(split, "r") as f:
                    split_dict = json.load(f)

            # Filter to p=23, m=3, standard (non attention-only)
            p = cfg_dict.get("p")
            m = cfg_dict.get("num_kv_pairs")
            arch = cfg_dict.get("architecture", "standard")
            if p != 23 or m != 3 or arch != "standard":
                continue

            found.append(
                {
                    "name": os.path.basename(path),
                    "out_dir": path,
                    "ckpt_path": ckpt_path,
                    "ckpt_kind": ckpt_kind,
                    "config": cfg_dict,
                    "split_metadata": split_dict,
                }
            )
    return found


# ---------------------------------------------------------------------------
# Test-set reconstruction (deterministic, matches scripts/run_hybrid_small_depth_test.py)
# ---------------------------------------------------------------------------

def build_test_indices(p: int, m: int, seed: int, train_frac: float, val_frac: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct the same exhaustive test partition the model was trained against."""
    inputs, labels = generate_hybrid_data_exhaustive(p=p, num_kv_pairs=m, seed=seed)
    n_total = inputs.shape[0]
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(n_total, generator=rng)

    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    test_idx = perm[n_train + n_val:]

    return inputs[test_idx], labels[test_idx], test_idx


# ---------------------------------------------------------------------------
# Sequence decoding
# ---------------------------------------------------------------------------

def decode_sequence(seq: torch.Tensor, p: int, m: int) -> dict[str, int]:
    """Decode a hybrid sequence into (v0, v1, v2, q1_idx, q2_idx)."""
    s = seq.tolist()
    key_offset = p
    kv_map: dict[int, int] = {}
    for slot in range(m):
        k_idx = s[2 * slot] - key_offset
        v_val = s[2 * slot + 1]
        kv_map[k_idx] = v_val
    q1_idx = s[2 * m + 1] - key_offset
    q2_idx = s[2 * m + 2] - key_offset
    out: dict[str, int] = {f"v{k}": kv_map[k] for k in range(m)}
    out["q1_idx"] = q1_idx
    out["q2_idx"] = q2_idx
    return out


def query_pair_label(q1_idx: int, q2_idx: int) -> str:
    a, b = sorted([q1_idx, q2_idx])
    return f"{a}{b}"


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_at_eq(model, inputs: torch.Tensor, device: torch.device, batch_size: int = 1024) -> torch.Tensor:
    """Return argmax predictions at the EQ position for every example."""
    preds: list[torch.Tensor] = []
    model.eval()
    for i in range(0, inputs.shape[0], batch_size):
        chunk = inputs[i : i + batch_size].to(device)
        logits = model(chunk)[:, -1, :]
        preds.append(logits.argmax(dim=-1).cpu())
    return torch.cat(preds, dim=0)


def build_counterfactual_sequences(seq: torch.Tensor, p: int, m: int) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Build the m*(m-1)/2 unordered counterfactual sequences sharing the same context."""
    base = seq.tolist()
    pairs = list(itertools.combinations(range(m), 2))  # (0,1), (0,2), (1,2)
    sequences: list[list[int]] = []
    for q1_idx, q2_idx in pairs:
        new = list(base)
        new[2 * m + 1] = p + q1_idx
        new[2 * m + 2] = p + q2_idx
        sequences.append(new)
    return torch.tensor(sequences, dtype=torch.long), pairs


# ---------------------------------------------------------------------------
# Model construction from checkpoint
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(ckpt_path: str, cfg_dict: dict, device: str):
    """Recreate the trained transformer architecture and load weights."""
    n_layers = int(cfg_dict["n_layers"])
    p = int(cfg_dict["p"])
    m = int(cfg_dict["num_kv_pairs"])
    seed = int(cfg_dict.get("seed", 0))

    model = create_standard_transformer(
        "hybrid_retrieve_add",
        cfg_overrides={
            "hybrid_p": p,
            "hybrid_num_kv_pairs": m,
            "n_layers": n_layers,
        },
        seed=seed,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt.get("epoch")


# ---------------------------------------------------------------------------
# Diagnostics for a single model
# ---------------------------------------------------------------------------

def diagnose_model(meta: dict[str, Any], device: torch.device, out_root: str) -> dict[str, Any]:
    cfg = meta["config"]
    p = int(cfg["p"])
    m = int(cfg["num_kv_pairs"])
    seed = int(cfg.get("seed", 0))
    train_frac = float(cfg.get("train_frac", 0.8))
    val_frac = float(cfg.get("val_frac", 0.1))
    n_layers = int(cfg["n_layers"])

    print(f"\n=== {meta['name']} (n_layers={n_layers}, ckpt={meta['ckpt_kind']}) ===")
    model, ckpt_epoch = load_model_from_checkpoint(meta["ckpt_path"], cfg, device)
    print(f"  loaded checkpoint: epoch={ckpt_epoch}")

    # --- test set ---
    test_inputs, test_labels, _ = build_test_indices(p, m, seed, train_frac, val_frac)
    print(f"  test examples: {test_inputs.shape[0]}")

    # --- predictions at EQ position ---
    preds = predict_at_eq(model, test_inputs, device)

    # --- decode every example and compute candidate sums ---
    rows: list[dict[str, Any]] = []
    for i in range(test_inputs.shape[0]):
        d = decode_sequence(test_inputs[i], p, m)
        v0, v1, v2 = d["v0"], d["v1"], d["v2"]
        s01 = (v0 + v1) % p
        s02 = (v0 + v2) % p
        s12 = (v1 + v2) % p
        pred = int(preds[i].item())
        true_label = int(test_labels[i].item())
        qpair = query_pair_label(d["q1_idx"], d["q2_idx"])
        rows.append(
            {
                "model": meta["name"],
                "n_layers": n_layers,
                "v0": v0,
                "v1": v1,
                "v2": v2,
                "q1_idx": d["q1_idx"],
                "q2_idx": d["q2_idx"],
                "query_pair": qpair,
                "true_label": true_label,
                "prediction": pred,
                "s01": s01,
                "s02": s02,
                "s12": s12,
                "pred_eq_correct": int(pred == true_label),
                "pred_eq_s01": int(pred == s01),
                "pred_eq_s02": int(pred == s02),
                "pred_eq_s12": int(pred == s12),
            }
        )

    # --- per-context counterfactual ---
    # Group rows by (v0,v1,v2) tuple for context-level analysis.
    seen_contexts: dict[tuple[int, int, int], int] = {}
    cf_rows: list[dict[str, Any]] = []
    base_cf_indices: list[int] = []
    for i, r in enumerate(rows):
        key = (r["v0"], r["v1"], r["v2"])
        if key not in seen_contexts:
            seen_contexts[key] = i
            base_cf_indices.append(i)

    n_cf = len(base_cf_indices)
    print(f"  unique contexts in test set: {n_cf}")

    # Build batched counterfactual inputs: for each context, 3 query variants.
    cf_seqs: list[torch.Tensor] = []
    cf_pairs_per_ctx: list[list[tuple[int, int]]] = []
    for idx in base_cf_indices:
        seqs, pairs = build_counterfactual_sequences(test_inputs[idx], p, m)
        cf_seqs.append(seqs)
        cf_pairs_per_ctx.append(pairs)
    cf_seqs_cat = torch.cat(cf_seqs, dim=0)
    cf_preds = predict_at_eq(model, cf_seqs_cat, device)

    cf_unique_count: list[int] = []
    cf_constant: list[int] = []
    cf_match_all: list[int] = []
    cf_correct: list[int] = []
    cf_correct_total = 0
    cf_correct_n = 0
    for c, idx in enumerate(base_cf_indices):
        r = rows[idx]
        s01, s02, s12 = r["s01"], r["s02"], r["s12"]
        targets_by_pair = {(0, 1): s01, (0, 2): s02, (1, 2): s12}
        preds_for_ctx = cf_preds[3 * c : 3 * c + 3].tolist()
        unique_n = len(set(preds_for_ctx))
        constant = int(unique_n == 1)
        match_all = int(
            preds_for_ctx[0] == s01
            and preds_for_ctx[1] == s02
            and preds_for_ctx[2] == s12
        )
        correct_per_q = [int(preds_for_ctx[i] == list(targets_by_pair.values())[i]) for i in range(3)]
        cf_correct_total += sum(correct_per_q)
        cf_correct_n += 3

        cf_unique_count.append(unique_n)
        cf_constant.append(constant)
        cf_match_all.append(match_all)
        cf_correct.append(sum(correct_per_q))
        cf_rows.append(
            {
                "model": meta["name"],
                "v0": r["v0"],
                "v1": r["v1"],
                "v2": r["v2"],
                "pred_q01": preds_for_ctx[0],
                "pred_q02": preds_for_ctx[1],
                "pred_q12": preds_for_ctx[2],
                "s01": s01,
                "s02": s02,
                "s12": s12,
                "n_unique_predictions": unique_n,
                "constant_prediction": constant,
                "matches_all_three": match_all,
                "n_correct_of_3": sum(correct_per_q),
            }
        )

    counterfactual_summary = {
        "model": meta["name"],
        "n_contexts": n_cf,
        "counterfactual_accuracy": cf_correct_total / max(cf_correct_n, 1),
        "mean_num_unique_predictions_per_context": float(np.mean(cf_unique_count)),
        "fraction_contexts_with_constant_prediction": float(np.mean(cf_constant)),
        "fraction_contexts_where_predictions_match_all_three_candidate_sums": float(np.mean(cf_match_all)),
    }

    # --- Aggregations ---
    # Accuracy by query pair
    qpairs = ["01", "02", "12"]
    by_qpair: list[dict[str, Any]] = []
    for qp in qpairs:
        rs = [r for r in rows if r["query_pair"] == qp]
        n = len(rs)
        if n == 0:
            continue
        acc = float(np.mean([r["pred_eq_correct"] for r in rs]))
        by_qpair.append(
            {"model": meta["name"], "query_pair": qp, "n_examples": n, "accuracy": acc}
        )

    # Candidate-sum match rates
    match_rates = {
        "s01": float(np.mean([r["pred_eq_s01"] for r in rows])),
        "s02": float(np.mean([r["pred_eq_s02"] for r in rows])),
        "s12": float(np.mean([r["pred_eq_s12"] for r in rows])),
        "correct_sum": float(np.mean([r["pred_eq_correct"] for r in rows])),
    }

    # Confusion matrix: row = actual query pair, col = matches s01/s02/s12/none
    confusion = np.zeros((3, 4), dtype=np.float64)
    for r in rows:
        ridx = qpairs.index(r["query_pair"])
        cols = [r["pred_eq_s01"], r["pred_eq_s02"], r["pred_eq_s12"]]
        if any(cols):
            cidx = cols.index(1)
        else:
            cidx = 3
        confusion[ridx, cidx] += 1
    # row-normalize
    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_norm = np.divide(confusion, row_sums, where=row_sums > 0)

    # Shortcut baselines
    baselines = {
        "always_s01": float(np.mean([int(r["s01"] == r["true_label"]) for r in rows])),
        "always_s02": float(np.mean([int(r["s02"] == r["true_label"]) for r in rows])),
        "always_s12": float(np.mean([int(r["s12"] == r["true_label"]) for r in rows])),
    }
    best_baseline_name = max(baselines, key=baselines.get)
    best_baseline_acc = baselines[best_baseline_name]

    overall_acc = match_rates["correct_sum"]

    # Verdict heuristic
    qpair_accs = {row["query_pair"]: row["accuracy"] for row in by_qpair}
    qpair_max = max(qpair_accs.values()) if qpair_accs else 0.0
    qpair_min = min(qpair_accs.values()) if qpair_accs else 0.0
    qpair_spread = qpair_max - qpair_min
    cf_acc = counterfactual_summary["counterfactual_accuracy"]
    constant_frac = counterfactual_summary["fraction_contexts_with_constant_prediction"]

    if cf_acc >= 0.85:
        verdict = "LIKELY_QUERY_CONDITIONED"
    elif constant_frac >= 0.5 or qpair_spread >= 0.20 or any(rate >= 0.40 for rate in [match_rates["s01"], match_rates["s02"], match_rates["s12"]] if rate > overall_acc + 0.05):
        verdict = "LIKELY_FIXED_PAIR_SHORTCUT"
    elif cf_acc <= 0.5 and constant_frac >= 0.2:
        verdict = "LIKELY_FIXED_PAIR_SHORTCUT"
    elif cf_acc >= 0.6 and constant_frac < 0.1:
        verdict = "LIKELY_QUERY_CONDITIONED"
    else:
        verdict = "AMBIGUOUS"

    summary = {
        "model": meta["name"],
        "n_layers": n_layers,
        "ckpt_kind": meta["ckpt_kind"],
        "ckpt_epoch": ckpt_epoch,
        "test_n": int(test_inputs.shape[0]),
        "overall_accuracy": overall_acc,
        "best_fixed_pair_baseline": best_baseline_name,
        "best_fixed_pair_baseline_acc": best_baseline_acc,
        "qpair_accuracy_spread": qpair_spread,
        "counterfactual_accuracy": cf_acc,
        "fraction_constant_prediction_across_queries": constant_frac,
        "verdict": verdict,
        "match_rate_s01": match_rates["s01"],
        "match_rate_s02": match_rates["s02"],
        "match_rate_s12": match_rates["s12"],
    }

    return {
        "rows": rows,
        "by_qpair": by_qpair,
        "match_rates": match_rates,
        "confusion": confusion,
        "confusion_norm": confusion_norm,
        "baselines": baselines,
        "counterfactual_summary": counterfactual_summary,
        "counterfactual_rows": cf_rows,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_accuracy_by_query_pair(per_model: list[dict], plots_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    qpairs = ["01", "02", "12"]
    width = 0.8 / max(len(per_model), 1)
    x = np.arange(len(qpairs))
    for i, m in enumerate(per_model):
        vals = []
        for qp in qpairs:
            row = next((r for r in m["by_qpair"] if r["query_pair"] == qp), None)
            vals.append(row["accuracy"] if row else 0.0)
        ax.bar(x + i * width, vals, width=width, label=m["summary"]["model"])
    ax.axhline(1 / 23, color="red", linestyle=":", label="chance (1/23)")
    ax.set_xticks(x + width * (len(per_model) - 1) / 2)
    ax.set_xticklabels([f"q={qp}" for qp in qpairs])
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy by query pair")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "accuracy_by_query_pair.png"), dpi=150)
    plt.close(fig)


def plot_match_rates(per_model: list[dict], plots_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cats = ["s01", "s02", "s12", "correct_sum"]
    width = 0.8 / max(len(per_model), 1)
    x = np.arange(len(cats))
    for i, m in enumerate(per_model):
        vals = [m["match_rates"][c] for c in cats]
        ax.bar(x + i * width, vals, width=width, label=m["summary"]["model"])
    ax.set_xticks(x + width * (len(per_model) - 1) / 2)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Pr[prediction == candidate]")
    ax.set_title("Candidate-sum match rates")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "candidate_match_rates.png"), dpi=150)
    plt.close(fig)


def plot_confusion(per_model: list[dict], plots_dir: str) -> None:
    n = len(per_model)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    qpairs = ["01", "02", "12"]
    cols = ["pred=s01", "pred=s02", "pred=s12", "pred=none"]
    for ax, m in zip(axes, per_model):
        im = ax.imshow(m["confusion_norm"], vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(4))
        ax.set_xticklabels(cols, rotation=30, ha="right")
        ax.set_yticks(range(3))
        ax.set_yticklabels([f"actual={qp}" for qp in qpairs])
        ax.set_title(m["summary"]["model"], fontsize=8)
        for i in range(3):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    f"{m['confusion_norm'][i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if m["confusion_norm"][i, j] < 0.5 else "black",
                    fontsize=8,
                )
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "candidate_confusion_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_counterfactual(per_model: list[dict], plots_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [m["summary"]["model"] for m in per_model]
    means = [m["counterfactual_summary"]["mean_num_unique_predictions_per_context"] for m in per_model]
    constant_frac = [m["counterfactual_summary"]["fraction_contexts_with_constant_prediction"] for m in per_model]
    cf_acc = [m["counterfactual_summary"]["counterfactual_accuracy"] for m in per_model]
    x = np.arange(len(names))
    w = 0.25
    ax.bar(x - w, means, width=w, label="mean unique preds (max=3)")
    ax.bar(x, constant_frac, width=w, label="frac. contexts w/ constant pred")
    ax.bar(x + w, cf_acc, width=w, label="counterfactual accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
    ax.set_title("Counterfactual query test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "counterfactual_unique_predictions.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_report(per_model: list[dict], out_dir: str) -> None:
    lines: list[str] = []
    lines.append("# Hybrid Query Diagnostics Report\n")

    lines.append("## 1. Executive summary\n")
    for m in per_model:
        s = m["summary"]
        lines.append(
            f"- **{s['model']}** (n_layers={s['n_layers']}, ckpt={s['ckpt_kind']}@epoch={s['ckpt_epoch']}): "
            f"acc={s['overall_accuracy']:.3f}, counterfactual_acc={s['counterfactual_accuracy']:.3f}, "
            f"verdict=**{s['verdict']}**"
        )
    lines.append("")

    lines.append("## 2. Models inspected\n")
    lines.append("| Model | n_layers | ckpt | epoch | test_n |")
    lines.append("|---|---:|---|---:|---:|")
    for m in per_model:
        s = m["summary"]
        lines.append(
            f"| {s['model']} | {s['n_layers']} | {s['ckpt_kind']} | {s['ckpt_epoch']} | {s['test_n']} |"
        )
    lines.append("")

    lines.append("## 3. Accuracy by query pair\n")
    lines.append("| Model | q=01 | q=02 | q=12 | spread |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in per_model:
        s = m["summary"]
        accs = {row["query_pair"]: row["accuracy"] for row in m["by_qpair"]}
        lines.append(
            f"| {s['model']} | {accs.get('01', 0):.3f} | {accs.get('02', 0):.3f} | "
            f"{accs.get('12', 0):.3f} | {s['qpair_accuracy_spread']:.3f} |"
        )
    lines.append("\n![accuracy_by_query_pair](plots/accuracy_by_query_pair.png)\n")

    lines.append("## 4. Candidate-sum match rates\n")
    lines.append("| Model | Pr[pred=s01] | Pr[pred=s02] | Pr[pred=s12] | Pr[pred=correct] |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in per_model:
        s = m["summary"]
        mr = m["match_rates"]
        lines.append(
            f"| {s['model']} | {mr['s01']:.3f} | {mr['s02']:.3f} | {mr['s12']:.3f} | {mr['correct_sum']:.3f} |"
        )
    lines.append("\n![candidate_match_rates](plots/candidate_match_rates.png)\n")

    lines.append("## 5. Counterfactual query test\n")
    lines.append("| Model | counterfactual acc | mean unique preds | frac constant | frac match-all-3 |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in per_model:
        s = m["summary"]
        cf = m["counterfactual_summary"]
        lines.append(
            f"| {s['model']} | {cf['counterfactual_accuracy']:.3f} | "
            f"{cf['mean_num_unique_predictions_per_context']:.3f} | "
            f"{cf['fraction_contexts_with_constant_prediction']:.3f} | "
            f"{cf['fraction_contexts_where_predictions_match_all_three_candidate_sums']:.3f} |"
        )
    lines.append("\n![counterfactual](plots/counterfactual_unique_predictions.png)\n")
    lines.append("\n![confusion_heatmap](plots/candidate_confusion_heatmap.png)\n")

    lines.append("## 6. Shortcut baseline comparison\n")
    lines.append("Always-predict baselines (no learning):\n")
    lines.append("| Model | always_s01 | always_s02 | always_s12 | best_baseline | model_acc |")
    lines.append("|---|---:|---:|---:|---|---:|")
    for m in per_model:
        s = m["summary"]
        b = m["baselines"]
        lines.append(
            f"| {s['model']} | {b['always_s01']:.3f} | {b['always_s02']:.3f} | {b['always_s12']:.3f} "
            f"| {s['best_fixed_pair_baseline']} ({s['best_fixed_pair_baseline_acc']:.3f}) | {s['overall_accuracy']:.3f} |"
        )
    lines.append("")

    lines.append("## 7. Interpretation\n")
    for m in per_model:
        s = m["summary"]
        lines.append(f"- **{s['model']}**: {s['verdict']}")
    lines.append("")
    lines.append(
        "Heuristic: `LIKELY_QUERY_CONDITIONED` if counterfactual accuracy ≥ 0.85; "
        "`LIKELY_FIXED_PAIR_SHORTCUT` if predictions are constant across queries for a large fraction of contexts, "
        "or if the model's accuracy is dominated by one fixed-pair baseline; otherwise `AMBIGUOUS`."
    )
    lines.append("")

    lines.append("## 8. Recommended next action\n")
    verdicts = [m["summary"]["verdict"] for m in per_model]
    if any(v == "LIKELY_FIXED_PAIR_SHORTCUT" for v in verdicts):
        lines.append(
            "At least one model shows shortcut behaviour. **Run Phase B**: train with auxiliary "
            "retrieval supervision (alpha=0.5) to encourage true query-conditioned retrieval."
        )
    elif all(v == "LIKELY_QUERY_CONDITIONED" for v in verdicts):
        lines.append(
            "Models appear query-conditioned but accuracy is low. Tune optimisation/capacity "
            "(weight decay, learning rate, depth, training length) before adding auxiliary heads."
        )
    else:
        lines.append("Mixed/ambiguous results. Phase B (auxiliary retrieval supervision) is still the recommended targeted intervention before further sweeping.")
    lines.append("")

    report_path = os.path.join(out_dir, "hybrid_query_diagnostics_report_for_chatgpt.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote report: {report_path}")


# ---------------------------------------------------------------------------
# CSV writing helpers
# ---------------------------------------------------------------------------

def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = PROJECT_ROOT
    out_dir = os.path.join(project_root, "outputs", "hybrid_query_diagnostics")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    metas = discover_checkpoints(project_root)
    if not metas:
        print("No p=23 m=3 standard hybrid checkpoints found.")
        return
    print(f"Found {len(metas)} checkpoint(s):")
    for meta in metas:
        print(f"  - {meta['name']} (ckpt={meta['ckpt_kind']})")

    per_model: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    by_qpair_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    cf_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for meta in metas:
        try:
            res = diagnose_model(meta, device, out_dir)
        except Exception as e:
            print(f"  ERROR diagnosing {meta['name']}: {e}")
            import traceback

            traceback.print_exc()
            continue

        per_model.append(res)
        all_rows.extend(res["rows"])
        by_qpair_rows.extend(res["by_qpair"])
        for k, v in res["match_rates"].items():
            candidate_rows.append({"model": meta["name"], "candidate": k, "match_rate": v})
        # confusion rows (one per (model, actual) row)
        qpairs = ["01", "02", "12"]
        col_names = ["pred_eq_s01", "pred_eq_s02", "pred_eq_s12", "pred_eq_none"]
        for ri, qp in enumerate(qpairs):
            row = {"model": meta["name"], "actual_query_pair": qp}
            for ci, cname in enumerate(col_names):
                row[cname] = float(res["confusion_norm"][ri, ci])
            confusion_rows.append(row)
        cf_rows.extend(res["counterfactual_rows"])
        summary_rows.append(res["summary"])

    # Write CSVs
    write_csv(os.path.join(out_dir, "diagnostics_all_examples.csv"), all_rows)
    write_csv(os.path.join(out_dir, "diagnostics_by_model.csv"), summary_rows)
    write_csv(os.path.join(out_dir, "accuracy_by_query_pair.csv"), by_qpair_rows)
    write_csv(os.path.join(out_dir, "candidate_match_rates.csv"), candidate_rows)
    write_csv(os.path.join(out_dir, "candidate_confusion_matrix.csv"), confusion_rows)
    write_csv(os.path.join(out_dir, "counterfactual_query_results.csv"), cf_rows)

    # Plots
    if per_model:
        plot_accuracy_by_query_pair(per_model, plots_dir)
        plot_match_rates(per_model, plots_dir)
        plot_confusion(per_model, plots_dir)
        plot_counterfactual(per_model, plots_dir)

    # Report
    write_report(per_model, out_dir)

    # Status
    with open(os.path.join(out_dir, "status.txt"), "w") as f:
        f.write("COMPLETED\n")

    print("\nPhase A diagnostics complete.")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
