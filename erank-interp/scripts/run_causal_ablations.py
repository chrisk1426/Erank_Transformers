"""
scripts/run_causal_ablations.py

Causal ablation study for completed standard checkpoints.

For each standard model (key_value, modular_addition, seeds 0-2), runs four
single-component ablations by zeroing each sublayer's output contribution to
the residual stream:
  - layer 0 attention
  - layer 0 MLP
  - layer 1 attention
  - layer 1 MLP

Measures val and test accuracy drop relative to the unablated base model.

Outputs:
  outputs/causal_ablation_core/ablation_all_runs.csv
  outputs/causal_ablation_core/ablation_summary_by_task.csv
  outputs/causal_ablation_core/plots/
  outputs/causal_ablation_core/ablation_report_for_chatgpt.md
"""

from __future__ import annotations

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
import torch
import torch.nn as nn
import yaml

from data.splits import (
    get_key_value_dataloaders_3way,
    get_modular_addition_dataloaders_3way,
)
from models.transformer import create_standard_transformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR    = os.path.join(PROJECT_ROOT, "outputs", "thesis_additions")
OUT_DIR     = os.path.join(PROJECT_ROOT, "outputs", "causal_ablation_core")
PLOTS_DIR   = os.path.join(OUT_DIR, "plots")
SEEDS       = [0, 1, 2]
TASKS       = ["key_value", "modular_addition"]
ABLATIONS   = [
    ("layer0", "attn", 0, "attn"),
    ("layer0", "mlp",  0, "mlp"),
    ("layer1", "attn", 1, "attn"),
    ("layer1", "mlp",  1, "mlp"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pred_position(task_name: str, n_ctx: int) -> int:
    if task_name == "modular_addition":
        return 2
    return n_ctx - 1


def _get_logits_at_pred(logits: torch.Tensor, task_name: str) -> torch.Tensor:
    if task_name == "modular_addition":
        return logits[:, 2, :]
    return logits[:, -1, :]


def _eval_accuracy(model, loader, task_name: str, device: torch.device,
                   fwd_hooks=None) -> float:
    model.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if fwd_hooks:
                logits = model.run_with_hooks(inputs, fwd_hooks=fwd_hooks)
            else:
                logits = model(inputs)
            pred = _get_logits_at_pred(logits, task_name).argmax(dim=-1)
            n_correct += (pred == labels).sum().item()
            n_total   += labels.size(0)
    return n_correct / n_total if n_total > 0 else 0.0


def _make_zero_hook(layer_idx: int, component: str):
    """Return a (hook_name, hook_fn) pair that zeros the specified component."""
    if component == "attn":
        name = f"blocks.{layer_idx}.hook_attn_out"
    else:
        name = f"blocks.{layer_idx}.hook_mlp_out"

    def hook_fn(value, hook):
        return torch.zeros_like(value)

    return name, hook_fn


def _build_dataloaders(task_name: str, cfg: dict, seed: int):
    if task_name == "key_value":
        return get_key_value_dataloaders_3way(
            num_keys=cfg["kv_num_keys"],
            vocab_size=cfg["kv_vocab_size"],
            train_frac=cfg["kv_train_frac"],
            batch_size=cfg["batch_size"],
            seed=seed,
        )
    elif task_name == "modular_addition":
        return get_modular_addition_dataloaders_3way(
            p=cfg["mod_p"],
            train_frac=cfg["mod_train_frac"],
            batch_size=cfg["batch_size"],
            seed=seed,
        )
    raise ValueError(task_name)


def _load_cfg():
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_cfg()
    print(f"Device: {device}")

    rows = []

    for task in TASKS:
        print(f"\n{'='*60}")
        print(f"  Task: {task}")
        print(f"{'='*60}")
        for seed in SEEDS:
            ckpt_path = os.path.join(
                BASE_DIR, f"seed_{seed}", f"standard_{task}",
                "checkpoints", "model_best_val.pt"
            )
            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] checkpoint not found: {ckpt_path}")
                continue

            print(f"\n  seed={seed}")

            # Build model and load weights
            if task == "key_value":
                overrides = {"kv_num_keys": cfg["kv_num_keys"],
                             "kv_vocab_size": cfg["kv_vocab_size"]}
            else:
                overrides = {"mod_p": cfg["mod_p"]}

            model = create_standard_transformer(task, cfg_overrides=overrides, seed=seed)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device)
            model.eval()

            # Build data loaders
            train_loader, val_loader, test_loader, _ = _build_dataloaders(task, cfg, seed)

            # Base accuracy
            base_val  = _eval_accuracy(model, val_loader,  task, device)
            base_test = _eval_accuracy(model, test_loader, task, device)
            print(f"    base  val_acc={base_val:.4f}  test_acc={base_test:.4f}")

            base_row = {
                "task": task, "seed": seed, "ablation": "none",
                "layer": -1, "component": "none",
                "val_acc": base_val,  "test_acc": base_test,
                "delta_val_acc": 0.0, "delta_test_acc": 0.0,
            }
            rows.append(base_row)

            # Ablations
            for abl_name, comp_short, layer_idx, component in ABLATIONS:
                hook_name, hook_fn = _make_zero_hook(layer_idx, component)

                # Skip MLP ablation if model has no MLP (shouldn't happen here)
                if component == "mlp" and getattr(model.cfg, "attn_only", False):
                    continue

                abl_val  = _eval_accuracy(model, val_loader,  task, device,
                                          fwd_hooks=[(hook_name, hook_fn)])
                abl_test = _eval_accuracy(model, test_loader, task, device,
                                          fwd_hooks=[(hook_name, hook_fn)])

                dv = base_val  - abl_val
                dt = base_test - abl_test
                print(f"    {abl_name}_{comp_short:4s}  val_acc={abl_val:.4f}  "
                      f"test_acc={abl_test:.4f}  Δval={dv:+.4f}  Δtest={dt:+.4f}")

                rows.append({
                    "task": task, "seed": seed,
                    "ablation": f"{abl_name}_{component}",
                    "layer": layer_idx, "component": component,
                    "val_acc": abl_val,  "test_acc": abl_test,
                    "delta_val_acc": dv,  "delta_test_acc": dt,
                })

    # -----------------------------------------------------------------------
    # Save full CSV
    # -----------------------------------------------------------------------
    all_csv = os.path.join(OUT_DIR, "ablation_all_runs.csv")
    fieldnames = ["task", "seed", "ablation", "layer", "component",
                  "val_acc", "test_acc", "delta_val_acc", "delta_test_acc"]
    with open(all_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {all_csv}")

    # -----------------------------------------------------------------------
    # Summary: mean ± std across seeds
    # -----------------------------------------------------------------------
    summary_rows = []
    for task in TASKS:
        for abl_name, comp_short, layer_idx, component in ABLATIONS:
            abl_key = f"layer{layer_idx}_{component}"
            base_tests = [r["test_acc"] for r in rows
                          if r["task"] == task and r["ablation"] == "none"]
            abl_tests  = [r["delta_test_acc"] for r in rows
                          if r["task"] == task and r["ablation"] == abl_key]
            if not abl_tests:
                continue
            summary_rows.append({
                "task": task,
                "ablation": abl_key,
                "layer": layer_idx,
                "component": component,
                "mean_base_test_acc":  float(np.mean(base_tests)),
                "mean_delta_test_acc": float(np.mean(abl_tests)),
                "std_delta_test_acc":  float(np.std(abl_tests)),
                "mean_delta_val_acc":  float(np.mean([r["delta_val_acc"] for r in rows
                                                      if r["task"] == task and r["ablation"] == abl_key])),
                "std_delta_val_acc":   float(np.std([r["delta_val_acc"] for r in rows
                                                     if r["task"] == task and r["ablation"] == abl_key])),
            })

    summary_csv = os.path.join(OUT_DIR, "ablation_summary_by_task.csv")
    sum_fields = ["task", "ablation", "layer", "component",
                  "mean_base_test_acc", "mean_delta_test_acc", "std_delta_test_acc",
                  "mean_delta_val_acc", "std_delta_val_acc"]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sum_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved: {summary_csv}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    _make_ablation_plots(rows, summary_rows, PLOTS_DIR, TASKS)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    _write_report(rows, summary_rows, OUT_DIR)
    print(f"\nDone. Output dir: {OUT_DIR}")


def _make_ablation_plots(rows, summary_rows, plots_dir, tasks):
    abl_labels = ["L0 attn", "L0 MLP", "L1 attn", "L1 MLP"]
    abl_keys   = ["layer0_attn", "layer0_mlp", "layer1_attn", "layer1_mlp"]
    colors     = ["#2196F3", "#FF9800", "#1565C0", "#E65100"]

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5), sharey=False)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        means = []
        stds  = []
        for abl_key in abl_keys:
            match = [r for r in summary_rows if r["task"] == task and r["ablation"] == abl_key]
            if match:
                means.append(match[0]["mean_delta_test_acc"])
                stds.append(match[0]["std_delta_test_acc"])
            else:
                means.append(0.0)
                stds.append(0.0)

        x = np.arange(len(abl_labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(abl_labels, fontsize=10)
        ax.set_ylabel("Δ test accuracy (base − ablated)", fontsize=10)
        ax.set_title(f"{task}\n(mean ± std, n=3 seeds)", fontsize=11)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylim(bottom=min(min(means) - 0.05, -0.05))

    fig.suptitle("Causal Ablation: Accuracy Drop by Sublayer", fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(plots_dir, "ablation_delta_test_acc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")

    # Per-seed detail plot
    for task in tasks:
        fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
        for ax, (abl_key, abl_label) in zip(axes, zip(abl_keys, abl_labels)):
            per_seed = [(r["seed"], r["delta_test_acc"]) for r in rows
                        if r["task"] == task and r["ablation"] == abl_key]
            per_seed.sort()
            seeds_  = [s for s, _ in per_seed]
            deltas_ = [d for _, d in per_seed]
            ax.bar(seeds_, deltas_, color="#2196F3", alpha=0.8)
            ax.set_title(abl_label, fontsize=10)
            ax.set_xlabel("Seed")
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            if ax == axes[0]:
                ax.set_ylabel("Δ test accuracy")
        fig.suptitle(f"{task} — per-seed ablation Δ test accuracy", fontsize=12)
        fig.tight_layout()
        path = os.path.join(plots_dir, f"ablation_per_seed_{task}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved: {path}")


def _write_report(rows, summary_rows, out_dir):
    lines = []
    a = lines.append

    a("# Causal Ablation Report")
    a("")
    a("## What was ablated")
    a("For each standard checkpoint (key_value and modular_addition, seeds 0–2), four")
    a("single-component ablations were run by zeroing the sublayer output:")
    a("- Layer 0 attention (`hook_attn_out`)")
    a("- Layer 0 MLP (`hook_mlp_out`)")
    a("- Layer 1 attention (`hook_attn_out`)")
    a("- Layer 1 MLP (`hook_mlp_out`)")
    a("")
    a("Δ accuracy = base accuracy − ablated accuracy. Positive = that component helps.")
    a("")

    # Summary table
    a("## Summary: mean Δ test accuracy across seeds")
    a("")
    a("| Task | Ablation | Mean base test acc | Mean Δ test acc | Std Δ test acc |")
    a("|---|---|---:|---:|---:|")
    for r in summary_rows:
        a(f"| {r['task']} | {r['ablation']} | {r['mean_base_test_acc']:.4f} | "
          f"{r['mean_delta_test_acc']:+.4f} | {r['std_delta_test_acc']:.4f} |")
    a("")

    # Per-run table
    a("## Per-run ablation results")
    a("")
    a("| Task | Seed | Ablation | Val acc | Test acc | Δ val | Δ test |")
    a("|---|---:|---|---:|---:|---:|---:|")
    for r in rows:
        a(f"| {r['task']} | {r['seed']} | {r['ablation']} | "
          f"{r['val_acc']:.4f} | {r['test_acc']:.4f} | "
          f"{r['delta_val_acc']:+.4f} | {r['delta_test_acc']:+.4f} |")
    a("")

    # Interpretation
    a("## Interpretation")
    a("")
    kv_attn  = [r for r in summary_rows if r["task"] == "key_value" and "attn" in r["ablation"]]
    kv_mlp   = [r for r in summary_rows if r["task"] == "key_value" and "mlp"  in r["ablation"]]
    mod_attn = [r for r in summary_rows if r["task"] == "modular_addition" and "attn" in r["ablation"]]
    mod_mlp  = [r for r in summary_rows if r["task"] == "modular_addition" and "mlp"  in r["ablation"]]

    kv_attn_l1  = next((r for r in kv_attn  if r["layer"] == 1), None)
    kv_mlp_l1   = next((r for r in kv_mlp   if r["layer"] == 1), None)
    mod_mlp_l1  = next((r for r in mod_mlp  if r["layer"] == 1), None)
    mod_attn_l1 = next((r for r in mod_attn if r["layer"] == 1), None)

    a("### Key-value retrieval")
    if kv_attn_l1 and kv_mlp_l1:
        a(f"- Layer 1 attn ablation Δ test = {kv_attn_l1['mean_delta_test_acc']:+.4f} "
          f"(std={kv_attn_l1['std_delta_test_acc']:.4f})")
        a(f"- Layer 1 MLP ablation Δ test  = {kv_mlp_l1['mean_delta_test_acc']:+.4f} "
          f"(std={kv_mlp_l1['std_delta_test_acc']:.4f})")
        if kv_attn_l1["mean_delta_test_acc"] > kv_mlp_l1["mean_delta_test_acc"]:
            a("- **Attention ablation hurts more than MLP ablation — consistent with attention-dominated routing.**")
        else:
            a("- MLP ablation hurts as much or more — unexpected for key-value.")
    a("")
    a("### Modular addition")
    if mod_mlp_l1 and mod_attn_l1:
        a(f"- Layer 1 MLP ablation Δ test  = {mod_mlp_l1['mean_delta_test_acc']:+.4f} "
          f"(std={mod_mlp_l1['std_delta_test_acc']:.4f})")
        a(f"- Layer 1 attn ablation Δ test = {mod_attn_l1['mean_delta_test_acc']:+.4f} "
          f"(std={mod_attn_l1['std_delta_test_acc']:.4f})")
        if mod_mlp_l1["mean_delta_test_acc"] > 0.01:
            a("- **Layer 1 MLP ablation causes meaningful accuracy drop — consistent with late-layer MLP contribution (eRank finding).**")
        else:
            a("- Layer 1 MLP ablation has small effect — may not fully agree with eRank finding.")
    a("")
    a("## Alignment with eRank")
    a("")
    a("eRank predicted:")
    a("- Key-value: Σ Δ_attn > 0, Σ Δ_mlp < 0 → attention does the work")
    a("- Modular addition: L1 Δ_mlp > 0 → late MLP contributes")
    a("")
    a("Causal ablation checks whether removing those components actually causes accuracy drops.")
    a("If ablation aligns with eRank, the eRank metric has causal validity.")

    report_path = os.path.join(out_dir, "ablation_report_for_chatgpt.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
