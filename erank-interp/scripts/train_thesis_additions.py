"""
scripts/train_thesis_additions.py

Orchestration script for thesis additions:
  - Required A: seed robustness (3 seeds × 4 existing task/arch combos)
  - Required B: hybrid retrieve-then-add (3 seeds × 2 arch combos)
  - Endpoint eRank for ALL selected checkpoints (including failed runs)

Seed policy: use fresh seeds [0, 1, 2] for all tasks to guarantee
a clean, compatible seed set.  Seed 42 from outputs/control_rerun/ is
excluded from the aggregate to avoid mixing pipelines.

Output structure:
  outputs/thesis_additions/
    seed_{N}/
      {model_type}_{task_name}/
        checkpoints/   best_val.pt, final.pt, summary.json
        results/       curves.json, training.png
        splits/        split_metadata.json
        erank/         endpoint_erank.json
        log.txt        (stdout mirror, written by run_thesis_additions.sh tee)
    aggregate/
      all_runs_summary.csv
      seed_robustness_summary.csv
      hybrid_summary.csv
      thesis_additions_report.md
      plots/           *.png
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from data.splits import (
    get_modular_addition_dataloaders_3way,
    get_key_value_dataloaders_3way,
    get_hybrid_dataloaders_3way,
)
from models.transformer import create_standard_transformer, create_attention_only_transformer
from training.train_v2 import train_v2
from analysis.extract_activations import extract_activations
from analysis.compute_erank_profiles import compute_erank_profile

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------

SEEDS = [0, 1, 2]

EXISTING_EXPERIMENTS = [
    ("standard",       "key_value"),
    ("standard",       "modular_addition"),
    ("attention_only", "key_value"),
    ("attention_only", "modular_addition"),
]

HYBRID_EXPERIMENTS = [
    ("standard",       "hybrid_retrieve_add"),
    ("attention_only", "hybrid_retrieve_add"),
]

ALL_EXPERIMENTS = EXISTING_EXPERIMENTS + HYBRID_EXPERIMENTS

VAL_FRAC_OF_TRAIN = 0.2
SUCCESS_THRESHOLD = 0.98

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "thesis_additions")

# eRank epochs: use a manageable schedule for all tasks
ERANK_EPOCHS_DENSE = [
    1, 5, 10, 25, 50, 100, 200, 300, 500, 750, 1000, 1250, 1500,
    1750, 2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000,
    15000, 20000, 30000, 40000, 50000,
]
ERANK_EPOCHS_LIGHT = [1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with open(os.path.join(PROJECT_ROOT, "configs", "default.yaml")) as f:
        return yaml.safe_load(f)


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _run_dir(seed: int, model_type: str, task_name: str) -> str:
    return os.path.join(OUTPUT_ROOT, f"seed_{seed}", f"{model_type}_{task_name}")


def _build_dataloaders(task: str, cfg: dict, seed: int) -> tuple:
    """Return (train_loader, val_loader, test_loader, split_meta)."""
    overridden = {**cfg, "seed": seed}
    if task == "modular_addition":
        return get_modular_addition_dataloaders_3way(
            p=overridden["mod_p"],
            train_frac=overridden["mod_train_frac"],
            val_frac_of_train=VAL_FRAC_OF_TRAIN,
            batch_size=overridden["batch_size"],
            seed=seed,
        )
    elif task == "key_value":
        return get_key_value_dataloaders_3way(
            num_keys=overridden["kv_num_keys"],
            vocab_size=overridden["kv_vocab_size"],
            train_frac=overridden["kv_train_frac"],
            val_frac_of_train=VAL_FRAC_OF_TRAIN,
            batch_size=overridden["batch_size"],
            seed=seed,
        )
    elif task == "hybrid_retrieve_add":
        return get_hybrid_dataloaders_3way(
            p=overridden["hybrid_p"],
            num_kv_pairs=overridden["hybrid_num_kv_pairs"],
            num_examples=overridden["hybrid_num_examples"],
            train_frac=overridden["hybrid_train_frac"],
            val_frac_of_train=VAL_FRAC_OF_TRAIN,
            batch_size=overridden["batch_size"],
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown task: {task!r}")


def _build_model(model_type: str, task: str, cfg: dict, seed: int):
    if task == "modular_addition":
        overrides = {"mod_p": cfg["mod_p"]}
    elif task == "key_value":
        overrides = {"kv_num_keys": cfg["kv_num_keys"], "kv_vocab_size": cfg["kv_vocab_size"]}
    elif task == "hybrid_retrieve_add":
        overrides = {
            "hybrid_p": cfg["hybrid_p"],
            "hybrid_num_kv_pairs": cfg["hybrid_num_kv_pairs"],
        }
    else:
        raise ValueError(f"Unknown task: {task!r}")

    if model_type == "standard":
        return create_standard_transformer(task, cfg_overrides=overrides, seed=seed)
    else:
        return create_attention_only_transformer(task, cfg_overrides=overrides, seed=seed)


def _get_erank_epochs(task: str) -> list[int]:
    if task == "key_value":
        return ERANK_EPOCHS_LIGHT
    else:
        return ERANK_EPOCHS_DENSE


def _plot_curves(history: dict, title: str, save_path: str) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = history["epoch"]
    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"],   label="val")
    ax_loss.plot(epochs, history["test_loss"],  label="test", linestyle="--", alpha=0.5)
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("CE loss")
    ax_loss.set_title(f"{title} — Loss"); ax_loss.legend()
    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"],   label="val")
    ax_acc.plot(epochs, history["test_acc"],  label="test", linestyle="--", alpha=0.5)
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_title(f"{title} — Accuracy"); ax_acc.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _compute_endpoint_erank(
    model_type: str,
    task: str,
    cfg: dict,
    seed: int,
    checkpoint_path: str,
    test_loader,
    device: str,
) -> dict | None:
    """Load best-val checkpoint and compute endpoint eRank on test set."""
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        overrides: dict = {}
        if task == "modular_addition":
            overrides = {"mod_p": cfg["mod_p"]}
        elif task == "key_value":
            overrides = {"kv_num_keys": cfg["kv_num_keys"], "kv_vocab_size": cfg["kv_vocab_size"]}
        elif task == "hybrid_retrieve_add":
            overrides = {"hybrid_p": cfg["hybrid_p"], "hybrid_num_kv_pairs": cfg["hybrid_num_kv_pairs"]}

        if model_type == "standard":
            model = create_standard_transformer(task, cfg_overrides=overrides, seed=seed)
        else:
            model = create_attention_only_transformer(task, cfg_overrides=overrides, seed=seed)

        model.load_state_dict(ckpt["model_state_dict"])
        acts = extract_activations(model, test_loader, task, n_examples=500, device=device)
        profile = compute_erank_profile(acts, n_layers=model.cfg.n_layers)
        return profile
    except Exception as exc:
        print(f"  [endpoint eRank] WARNING: failed — {exc}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Single-run training
# ---------------------------------------------------------------------------

def run_one(
    model_type: str,
    task_name: str,
    seed: int,
    cfg: dict,
    device: str,
) -> dict:
    """Train one model/task/seed and save all artefacts.  Returns summary dict."""
    run_dir = _run_dir(seed, model_type, task_name)
    ckpt_dir    = os.path.join(run_dir, "checkpoints")
    results_dir = os.path.join(run_dir, "results")
    splits_dir  = os.path.join(run_dir, "splits")
    erank_dir   = os.path.join(run_dir, "erank")
    for d in [ckpt_dir, results_dir, splits_dir, erank_dir]:
        os.makedirs(d, exist_ok=True)

    run_name = f"seed{seed}_{model_type}_{task_name}"
    print(f"\n{'='*64}")
    print(f"  Run: {run_name}")
    print(f"{'='*64}")

    _seed_all(seed)

    # --- data ---
    train_loader, val_loader, test_loader, split_meta = _build_dataloaders(task_name, cfg, seed)
    split_path = os.path.join(splits_dir, "split_metadata.json")
    with open(split_path, "w") as f:
        json.dump(split_meta, f, indent=2)

    # --- model ---
    model = _build_model(model_type, task_name, cfg, seed)

    erank_epochs = _get_erank_epochs(task_name)
    erank_n = min(500, split_meta["n_val"])

    # --- train ---
    history = train_v2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=cfg["num_epochs"],
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        checkpoint_dir=ckpt_dir,
        checkpoint_name="model",
        device=device,
        task_name=task_name,
        erank_eval_epochs=erank_epochs,
        erank_n_examples=erank_n,
    )

    # --- augment checkpoints with metadata ---
    extra_fields = {
        "cfg": {**cfg, "seed": seed, "val_frac_of_train": VAL_FRAC_OF_TRAIN},
        "model_type": model_type,
        "task_name": task_name,
    }
    best_val_path = os.path.join(ckpt_dir, "model_best_val.pt")
    final_path    = os.path.join(ckpt_dir, "model_final.pt")
    for ckpt_path in [best_val_path, final_path]:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            ckpt.update(extra_fields)
            torch.save(ckpt, ckpt_path)

    # --- test accuracy at best-val epoch ---
    best_val_epoch = history["best_val_epoch"]
    if best_val_epoch is not None and best_val_epoch in history["epoch"]:
        idx = history["epoch"].index(best_val_epoch)
        test_acc_at_best_val = history["test_acc"][idx]
    else:
        test_acc_at_best_val = history["test_acc"][-1] if history["test_acc"] else None

    final_epoch    = history["epoch"][-1]
    final_test_acc = history["test_acc"][-1]

    # --- endpoint eRank (on best-val checkpoint, test set) ---
    endpoint_erank: dict | None = None
    if os.path.exists(best_val_path):
        endpoint_erank = _compute_endpoint_erank(
            model_type, task_name, cfg, seed, best_val_path, test_loader, device
        )
    erank_path = os.path.join(erank_dir, "endpoint_erank.json")
    n_layers = 2
    sum_delta_attn = None
    sum_delta_mlp  = None
    layer_deltas: dict = {}
    if endpoint_erank is not None:
        with open(erank_path, "w") as f:
            json.dump(endpoint_erank, f, indent=2)
        sum_delta_attn = sum(endpoint_erank["delta_erank_attn"].values())
        sum_delta_mlp  = sum(endpoint_erank["delta_erank_mlp"].values())
        for l in range(n_layers):
            lk = f"layer_{l}"
            layer_deltas[f"layer{l}_delta_attn"] = endpoint_erank["delta_erank_attn"].get(lk)
            layer_deltas[f"layer{l}_delta_mlp"]  = endpoint_erank["delta_erank_mlp"].get(lk)
        print(
            f"  Endpoint eRank: "
            + "  ".join(
                f"L{l}: pre={endpoint_erank['erank'][f'layer_{l}']['pre']:.2f} "
                f"mid={endpoint_erank['erank'][f'layer_{l}']['mid']:.2f} "
                f"post={endpoint_erank['erank'][f'layer_{l}']['post']:.2f}"
                for l in range(n_layers)
            )
        )
        print(f"  Σ Δ_attn={sum_delta_attn:.3f}   Σ Δ_mlp={sum_delta_mlp:.3f}")
    else:
        with open(erank_path, "w") as f:
            json.dump({"error": "endpoint eRank computation failed"}, f)

    # --- checkpoint summary ---
    is_success = (
        test_acc_at_best_val is not None and test_acc_at_best_val >= SUCCESS_THRESHOLD
    )
    notes = "success" if is_success else f"low_accuracy (test_acc={test_acc_at_best_val:.4f})"

    summary = {
        "run_id": run_name,
        "task": task_name,
        "architecture": model_type,
        "seed": seed,
        "learning_rate": cfg["learning_rate"],
        "weight_decay": cfg["weight_decay"],
        "batch_size": cfg["batch_size"],
        "max_epochs": cfg["num_epochs"],
        "stopping_epoch": final_epoch,
        "best_val_epoch": best_val_epoch,
        "best_val_acc": history["best_val_acc"],
        "test_acc": test_acc_at_best_val,
        "sum_delta_attn": sum_delta_attn,
        "sum_delta_mlp": sum_delta_mlp,
        **layer_deltas,
        "checkpoint_path": best_val_path,
        "erank_result_path": erank_path,
        "log_path": os.path.join(run_dir, "log.txt"),
        "notes": notes,
        "success": is_success,
    }
    summary_path = os.path.join(ckpt_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --- training curves ---
    curves_path = os.path.join(results_dir, "curves.json")
    with open(curves_path, "w") as f:
        json.dump(history, f)
    _plot_curves(history, title=run_name, save_path=os.path.join(results_dir, "training.png"))

    print(f"  Done. test_acc@best_val={test_acc_at_best_val}  notes={notes}")
    return summary


# ---------------------------------------------------------------------------
# Aggregate and report
# ---------------------------------------------------------------------------

def aggregate_results(all_summaries: list[dict], cfg: dict) -> None:
    """Compute mean ± std across seeds and write CSV + markdown report."""
    agg_dir = os.path.join(OUTPUT_ROOT, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)
    plots_dir = os.path.join(agg_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # --- all_runs_summary.csv ---
    if not all_summaries:
        print("  No summaries to aggregate.")
        return

    csv_path = os.path.join(agg_dir, "all_runs_summary.csv")
    fieldnames = [
        "run_id", "task", "architecture", "seed",
        "learning_rate", "weight_decay", "batch_size", "max_epochs",
        "stopping_epoch", "best_val_epoch", "best_val_acc", "test_acc",
        "sum_delta_attn", "sum_delta_mlp",
        "layer0_delta_attn", "layer0_delta_mlp",
        "layer1_delta_attn", "layer1_delta_mlp",
        "checkpoint_path", "erank_result_path", "log_path", "notes", "success",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"  All-runs summary: {csv_path}")

    # --- per-task-arch aggregation ---
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for s in all_summaries:
        groups[(s["task"], s["architecture"])].append(s)

    def _mean_std(vals: list) -> tuple[float | None, float | None]:
        cleaned = [v for v in vals if v is not None]
        if not cleaned:
            return None, None
        arr = np.array(cleaned, dtype=float)
        return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else (float(arr[0]), 0.0)

    agg_rows = []
    for (task, arch), runs in sorted(groups.items()):
        test_accs    = [r["test_acc"] for r in runs]
        sum_attns    = [r["sum_delta_attn"] for r in runs]
        sum_mlps     = [r["sum_delta_mlp"] for r in runs]
        l0_attns     = [r.get("layer0_delta_attn") for r in runs]
        l0_mlps      = [r.get("layer0_delta_mlp") for r in runs]
        l1_attns     = [r.get("layer1_delta_attn") for r in runs]
        l1_mlps      = [r.get("layer1_delta_mlp") for r in runs]
        n_success    = sum(1 for r in runs if r.get("success"))
        n_total      = len(runs)

        ma, sa   = _mean_std(test_accs)
        mda, sda = _mean_std(sum_attns)
        mdm, sdm = _mean_std(sum_mlps)
        ml0a, sl0a = _mean_std(l0_attns)
        ml0m, sl0m = _mean_std(l0_mlps)
        ml1a, sl1a = _mean_std(l1_attns)
        ml1m, sl1m = _mean_std(l1_mlps)

        agg_rows.append({
            "task": task, "architecture": arch,
            "mean_test_acc": ma, "std_test_acc": sa,
            "mean_sum_delta_attn": mda, "std_sum_delta_attn": sda,
            "mean_sum_delta_mlp": mdm, "std_sum_delta_mlp": sdm,
            "mean_layer0_delta_attn": ml0a, "std_layer0_delta_attn": sl0a,
            "mean_layer0_delta_mlp": ml0m, "std_layer0_delta_mlp": sl0m,
            "mean_layer1_delta_attn": ml1a, "std_layer1_delta_attn": sl1a,
            "mean_layer1_delta_mlp": ml1m, "std_layer1_delta_mlp": sl1m,
            "num_successful_runs": n_success, "num_total_runs": n_total,
            "seeds": str([r["seed"] for r in runs]),
        })

    agg_fields = [
        "task", "architecture",
        "mean_test_acc", "std_test_acc",
        "mean_sum_delta_attn", "std_sum_delta_attn",
        "mean_sum_delta_mlp", "std_sum_delta_mlp",
        "mean_layer0_delta_attn", "std_layer0_delta_attn",
        "mean_layer0_delta_mlp", "std_layer0_delta_mlp",
        "mean_layer1_delta_attn", "std_layer1_delta_attn",
        "mean_layer1_delta_mlp", "std_layer1_delta_mlp",
        "num_successful_runs", "num_total_runs", "seeds",
    ]

    # seed robustness summary (existing tasks)
    rob_path = os.path.join(agg_dir, "seed_robustness_summary.csv")
    rob_rows = [r for r in agg_rows if r["task"] != "hybrid_retrieve_add"]
    with open(rob_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rob_rows)
    print(f"  Seed robustness summary: {rob_path}")

    # hybrid summary
    hyb_path = os.path.join(agg_dir, "hybrid_summary.csv")
    hyb_rows = [r for r in agg_rows if r["task"] == "hybrid_retrieve_add"]
    with open(hyb_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(hyb_rows)
    print(f"  Hybrid summary: {hyb_path}")

    # --- plots ---
    _make_plots(all_summaries, agg_rows, plots_dir)

    # --- markdown report ---
    _write_report(all_summaries, agg_rows, agg_dir, cfg)


def _fmt(v, decimals: int = 4) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def _make_plots(all_summaries: list[dict], agg_rows: list[dict], plots_dir: str) -> None:
    tasks   = sorted(set(r["task"] for r in all_summaries))
    archs   = sorted(set(r["architecture"] for r in all_summaries))
    x_labels = [f"{t[:8]}\n{a[:7]}" for t in tasks for a in archs]
    x_pos = np.arange(len(x_labels))

    # --- 1. Accuracy bar chart by task × arch × seed ---
    fig, ax = plt.subplots(figsize=(max(8, len(x_labels)*1.2), 5))
    seeds = sorted(set(r["seed"] for r in all_summaries))
    width = 0.8 / max(len(seeds), 1)
    for si, seed in enumerate(seeds):
        accs = []
        for t in tasks:
            for a in archs:
                row = next((r for r in all_summaries if r["task"]==t and r["architecture"]==a and r["seed"]==seed), None)
                accs.append(row["test_acc"] if row and row["test_acc"] is not None else 0.0)
        offset = (si - len(seeds)/2 + 0.5) * width
        ax.bar(x_pos + offset, accs, width=width, label=f"seed {seed}", alpha=0.8)
    ax.axhline(1/113, color="gray", linestyle="--", linewidth=0.8, label="chance (1/113)")
    ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("Test accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Test accuracy by task × architecture × seed")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "accuracy_by_condition.png"), dpi=120)
    plt.close(fig)
    print("  Plot: accuracy_by_condition.png")

    # --- 2. Mean Σ Δ_attn and Σ Δ_mlp bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(x_labels)*1.5), 5))
    for ax, metric_key, title_str in [
        (axes[0], ("mean_sum_delta_attn", "std_sum_delta_attn"), "Mean Σ Δ_attn"),
        (axes[1], ("mean_sum_delta_mlp",  "std_sum_delta_mlp"),  "Mean Σ Δ_mlp"),
    ]:
        means, stds = [], []
        for t in tasks:
            for a in archs:
                row = next((r for r in agg_rows if r["task"]==t and r["architecture"]==a), None)
                means.append(row[metric_key[0]] if row and row[metric_key[0]] is not None else 0.0)
                stds.append(row[metric_key[1]] if row and row[metric_key[1]] is not None else 0.0)
        ax.bar(x_pos, means, yerr=stds, capsize=4, alpha=0.8)
        ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel("eRank delta"); ax.set_title(title_str); ax.axhline(0, color="k", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "mean_erank_deltas.png"), dpi=120)
    plt.close(fig)
    print("  Plot: mean_erank_deltas.png")

    # --- 3. Per-seed scatter for Σ Δ_attn and Σ Δ_mlp ---
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(x_labels)*1.5), 5))
    colors = plt.cm.tab10.colors
    for ax, key, title_str in [
        (axes[0], "sum_delta_attn", "Σ Δ_attn per seed"),
        (axes[1], "sum_delta_mlp",  "Σ Δ_mlp per seed"),
    ]:
        for si, seed in enumerate(seeds):
            vals = []
            for t in tasks:
                for a in archs:
                    row = next((r for r in all_summaries if r["task"]==t and r["architecture"]==a and r["seed"]==seed), None)
                    vals.append(row[key] if row and row.get(key) is not None else float("nan"))
            ax.scatter(x_pos, vals, label=f"seed {seed}", color=colors[si % len(colors)], zorder=3)
        ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel("eRank delta"); ax.set_title(title_str); ax.axhline(0, color="k", linewidth=0.5)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "perseed_erank_scatter.png"), dpi=120)
    plt.close(fig)
    print("  Plot: perseed_erank_scatter.png")

    # --- 4. Hybrid layerwise eRank deltas ---
    hybrid_summaries = [r for r in all_summaries if r["task"] == "hybrid_retrieve_add"]
    if hybrid_summaries:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        h_archs = sorted(set(r["architecture"] for r in hybrid_summaries))
        for ax, delta_key, title_str in [
            (axes[0], ("layer0_delta_attn", "layer1_delta_attn"), "Hybrid Δ_attn per layer"),
            (axes[1], ("layer0_delta_mlp",  "layer1_delta_mlp"),  "Hybrid Δ_mlp per layer"),
        ]:
            for ai, arch in enumerate(h_archs):
                rows = [r for r in hybrid_summaries if r["architecture"] == arch]
                for l, lkey in enumerate(delta_key):
                    vals = [r.get(lkey) for r in rows if r.get(lkey) is not None]
                    if vals:
                        ax.scatter(
                            [l + ai*0.2]*len(vals), vals,
                            label=f"{arch} L{l}" if l == 0 else None,
                            color=colors[ai % len(colors)], alpha=0.7,
                        )
                        ax.errorbar(
                            l + ai*0.2, np.mean(vals),
                            yerr=np.std(vals) if len(vals) > 1 else 0,
                            fmt="D", color=colors[ai % len(colors)],
                        )
            ax.set_xticks([0.1, 1.1]); ax.set_xticklabels(["Layer 0", "Layer 1"])
            ax.set_ylabel("eRank delta"); ax.set_title(title_str); ax.axhline(0, color="k", linewidth=0.5)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "hybrid_layerwise_erank.png"), dpi=120)
        plt.close(fig)
        print("  Plot: hybrid_layerwise_erank.png")


def _write_report(
    all_summaries: list[dict],
    agg_rows: list[dict],
    agg_dir: str,
    cfg: dict,
) -> None:
    import socket, subprocess
    hostname = socket.gethostname()
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", PROJECT_ROOT, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    tasks = sorted(set(r["task"] for r in all_summaries))
    archs = sorted(set(r["architecture"] for r in all_summaries))

    # Build per-run table rows
    table_lines = ["| run_id | task | arch | seed | test_acc | Σ Δ_attn | Σ Δ_mlp | notes |",
                   "|---|---|---|---|---|---|---|---|"]
    for s in sorted(all_summaries, key=lambda x: (x["task"], x["architecture"], x["seed"])):
        table_lines.append(
            f"| {s['run_id']} | {s['task']} | {s['architecture']} | {s['seed']} "
            f"| {_fmt(s.get('test_acc'))} | {_fmt(s.get('sum_delta_attn'))} "
            f"| {_fmt(s.get('sum_delta_mlp'))} | {s.get('notes','')} |"
        )

    # Aggregate table
    agg_lines = ["| task | arch | mean test_acc | std | mean Σ Δ_attn | std | mean Σ Δ_mlp | std | success/total |",
                 "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(agg_rows, key=lambda x: (x["task"], x["architecture"])):
        agg_lines.append(
            f"| {r['task']} | {r['architecture']} "
            f"| {_fmt(r.get('mean_test_acc'))} | {_fmt(r.get('std_test_acc'))} "
            f"| {_fmt(r.get('mean_sum_delta_attn'))} | {_fmt(r.get('std_sum_delta_attn'))} "
            f"| {_fmt(r.get('mean_sum_delta_mlp'))} | {_fmt(r.get('std_sum_delta_mlp'))} "
            f"| {r['num_successful_runs']}/{r['num_total_runs']} |"
        )

    report = f"""\
# Thesis Additions Results Report

## 1. What was implemented

- **Required A**: Seed robustness for existing tasks (key-value retrieval and modular
  addition) across seeds [0, 1, 2], for both standard and attention-only architectures.
- **Required B**: New hybrid retrieve-then-add task (y = (v(q1)+v(q2)) mod 113),
  trained with standard and attention-only architectures across seeds [0, 1, 2].
- Endpoint eRank computed for all selected checkpoints including failed/low-accuracy runs.
- Aggregate mean ± std tables for test accuracy and eRank deltas.

## 2. Repo / command details

- Git commit: `{git_commit}`
- Hostname: `{hostname}`
- Python: `{sys.version.split()[0]}`
- Environment: venv at `erank-interp/venv`
- tmux session: `thesis_additions`
- Output directory: `{OUTPUT_ROOT}`
- Main script: `scripts/train_thesis_additions.py`
- Run script: `scripts/run_thesis_additions.sh`

## 3. Experimental matrix

3 seeds × 3 tasks × 2 architectures = 18 runs total.

| Task | Architecture | Seeds |
|---|---|---|
| key_value | standard | 0, 1, 2 |
| key_value | attention_only | 0, 1, 2 |
| modular_addition | standard | 0, 1, 2 |
| modular_addition | attention_only | 0, 1, 2 |
| hybrid_retrieve_add | standard | 0, 1, 2 |
| hybrid_retrieve_add | attention_only | 0, 1, 2 |

## 4. Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | {cfg.get('learning_rate', 0.0003)} |
| Weight decay | {cfg.get('weight_decay', 1.0)} |
| Batch size | {cfg.get('batch_size', 256)} |
| Max epochs | {cfg.get('num_epochs', 50000)} |
| Early stop | val_acc > 0.98 for 500 consecutive epochs |
| Val fraction | {VAL_FRAC_OF_TRAIN} of train split |
| Seed set | [0, 1, 2] (fresh; seed 42 from control_rerun excluded) |

## 5. Dataset details

### Existing tasks

**Key-value retrieval**
- num_keys={cfg.get('kv_num_keys', 5)}, vocab_size={cfg.get('kv_vocab_size', 30)}, num_examples=10000
- Sequence: [k0,v0,...,k4,v4, QUERY_KEY]
- Train frac: {cfg.get('kv_train_frac', 0.7)}

**Modular addition**
- p={cfg.get('mod_p', 113)}, train_frac={cfg.get('mod_train_frac', 0.3)}, n_total=p²=12769
- Sequence: [a, b, EQ]
- Label: (a+b) mod p

### Hybrid retrieve-then-add task

- **p**: {cfg.get('hybrid_p', 113)}
- **num_kv_pairs**: {cfg.get('hybrid_num_kv_pairs', 4)}
- **num_examples**: {cfg.get('hybrid_num_examples', 20000)}
- **train_frac**: {cfg.get('hybrid_train_frac', 0.7)} → ~14000 train+val, ~6000 test
- **val_frac_of_train**: {VAL_FRAC_OF_TRAIN} → ~11200 train, ~2800 val
- **context-specific bindings**: yes — values re-sampled per example
- **q1 != q2 enforced**: yes — sampled without replacement
- **kv order randomised**: yes — permuted per example, seeded
- **sequence format**: [K_perm0, V_perm0, ..., K_perm3, V_perm3, QUERY, q1_key, q2_key, EQ]
- **d_vocab**: p + num_kv_pairs + 2 = {cfg.get('hybrid_p', 113) + cfg.get('hybrid_num_kv_pairs', 4) + 2}
- **label formula**: y = (v(q1) + v(q2)) mod {cfg.get('hybrid_p', 113)}
- **sanity test**: verified on 20 random val examples per seed (assertions in splits.py)
- **chance accuracy**: 1/{cfg.get('hybrid_p', 113)} ≈ {1/cfg.get('hybrid_p', 113):.5f}

Example:
```
Sequence: [K1:7, K0:42, K3:88, K2:15, QUERY, K1, K3, EQ]
Label: (7 + 88) % 113 = 95
```

## 6. Seed robustness results

Seed set: [0, 1, 2] (fresh; seed 42 from outputs/control_rerun/ excluded to avoid pipeline mixing)

### Per-run results (existing tasks only)

{chr(10).join(l for l in table_lines if any(t in l for t in ['run_id','---','key_value','modular']))}

### Aggregate mean ± std (existing tasks)

{chr(10).join(l for l in agg_lines if any(t in l for t in ['task','---','key_value','modular']))}

**Qualitative conclusions (to be filled by user after reviewing numbers above)**:
- Key-value attention-sufficiency: TBD
- Standard modular addition stability: TBD
- Attention-only modular addition stability: TBD
- Late-layer MLP contribution for modular addition: TBD

## 7. Hybrid task results

### Per-run results

{chr(10).join(l for l in table_lines if any(t in l for t in ['run_id','---','hybrid']))}

### Aggregate

{chr(10).join(l for l in agg_lines if any(t in l for t in ['task','---','hybrid']))}

**Note**: eRank measures spectral/geometric spread, not causal importance.
See plots/hybrid_layerwise_erank.png for layerwise delta profiles.

## 8. Optional hyperparameter robustness results

Not run in this batch (weight_decay=1.0 only).

## 9. Failures / caveats

- Any run with test_acc < {SUCCESS_THRESHOLD} is marked low_accuracy; endpoint eRank was still computed.
- Attention-only modular addition is expected to be weak/unstable under these matched hyperparameters.
- Seed 42 from outputs/control_rerun/ was excluded to avoid mixing incompatible pipeline versions.

## 10. Files produced

All outputs under: `{OUTPUT_ROOT}/`

Per run: `seed_{{N}}/{{model}}_{{task}}/checkpoints/`, `results/`, `splits/`, `erank/`
Aggregate: `aggregate/all_runs_summary.csv`, `seed_robustness_summary.csv`, `hybrid_summary.csv`
Plots: `aggregate/plots/*.png`

## 11. Suggested next steps

- Ablation study: freeze/remove MLP or attention in trained hybrid model
- Activation patching: patch hybrid model with key-value or modular-addition activations
- Optional Addition C: rerun with weight_decay=0.1 if compute allows
- If hybrid task did not learn: try num_kv_pairs=3 or num_kv_pairs=2
"""

    report_path = os.path.join(agg_dir, "thesis_additions_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = _load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")
    print(f"Experiments: {[f'{m}_{t}' for m,t in ALL_EXPERIMENTS]}")
    print(f"Output root: {OUTPUT_ROOT}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "aggregate"), exist_ok=True)

    all_summaries: list[dict] = []

    for seed in SEEDS:
        for model_type, task_name in ALL_EXPERIMENTS:
            try:
                summary = run_one(model_type, task_name, seed, cfg, device)
                all_summaries.append(summary)
            except Exception as exc:
                print(f"\n[ERROR] seed={seed} {model_type}_{task_name}: {exc}")
                traceback.print_exc()
                # Record failure row so it appears in aggregate
                run_dir = _run_dir(seed, model_type, task_name)
                all_summaries.append({
                    "run_id": f"seed{seed}_{model_type}_{task_name}",
                    "task": task_name,
                    "architecture": model_type,
                    "seed": seed,
                    "learning_rate": cfg.get("learning_rate"),
                    "weight_decay": cfg.get("weight_decay"),
                    "batch_size": cfg.get("batch_size"),
                    "max_epochs": cfg.get("num_epochs"),
                    "stopping_epoch": None,
                    "best_val_epoch": None,
                    "best_val_acc": None,
                    "test_acc": None,
                    "sum_delta_attn": None,
                    "sum_delta_mlp": None,
                    "checkpoint_path": None,
                    "erank_result_path": None,
                    "log_path": None,
                    "notes": f"CRASHED: {exc}",
                    "success": False,
                })
            finally:
                # Checkpoint partial results so far
                partial_csv = os.path.join(OUTPUT_ROOT, "aggregate", "partial_summary.csv")
                if all_summaries:
                    fields = list(all_summaries[0].keys())
                    with open(partial_csv, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                        writer.writeheader()
                        writer.writerows(all_summaries)

    print(f"\n{'='*64}")
    print(f"  All runs complete.  Aggregating results...")
    print(f"{'='*64}")
    aggregate_results(all_summaries, cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
