"""
scripts/run_graded_ablation_complete.py

Run any graded ablations missing from `outputs/graded_ablation_core/` so that
all four sublayers (L0/L1 x attn/MLP) have dose-response curves for both
modular_addition and key_value, then produce two clean figures showing
the dose-response curves for all four components per task.

Outputs:
  outputs/graded_ablation_complete/missing_runs.csv
  outputs/graded_ablation_complete/all_components_summary.csv
  outputs/graded_ablation_complete/modular_addition_partial_ablation.png
  outputs/graded_ablation_complete/key_value_partial_ablation.png
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
import torch
import torch.nn as nn
import yaml

from data.splits import (
    get_key_value_dataloaders_3way,
    get_modular_addition_dataloaders_3way,
)
from models.transformer import create_standard_transformer

BASE_DIR = os.path.join(PROJECT_ROOT, "outputs", "thesis_additions")
OUT_DIR  = os.path.join(PROJECT_ROOT, "outputs", "graded_ablation_complete")
EXISTING_CORE = os.path.join(PROJECT_ROOT, "outputs", "graded_ablation_core",
                             "graded_ablation_summary.csv")
EXISTING_L0_ATTN_KV = os.path.join(PROJECT_ROOT, "outputs", "graded_ablation_l0_attn",
                                   "kv_l0_attn_graded.csv")
SEEDS = [0, 1, 2]
ABLATION_STRENGTHS = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]

CHANCE = {"key_value": 1.0 / 30, "modular_addition": 1.0 / 113}

ALL_COMPONENTS = [
    ("key_value",        0, "attn"),
    ("key_value",        0, "mlp"),
    ("key_value",        1, "attn"),
    ("key_value",        1, "mlp"),
    ("modular_addition", 0, "attn"),
    ("modular_addition", 0, "mlp"),
    ("modular_addition", 1, "attn"),
    ("modular_addition", 1, "mlp"),
]


def _get_logits_at_pred(logits, task_name):
    if task_name == "modular_addition":
        return logits[:, 2, :]
    return logits[:, -1, :]


def _eval_accuracy(model, loader, task_name, device, fwd_hooks=None):
    model.eval()
    n_correct = n_total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device); labels = labels.to(device)
            if fwd_hooks:
                logits = model.run_with_hooks(inputs, fwd_hooks=fwd_hooks)
            else:
                logits = model(inputs)
            pred = _get_logits_at_pred(logits, task_name).argmax(dim=-1)
            n_correct += (pred == labels).sum().item()
            n_total   += labels.size(0)
    return n_correct / n_total if n_total > 0 else 0.0


def _scale_hook(layer_idx, component, scale):
    name = f"blocks.{layer_idx}.hook_{'attn_out' if component == 'attn' else 'mlp_out'}"
    def hook_fn(value, hook):
        return value * scale
    return name, hook_fn


def _build_loaders(task, cfg, seed):
    if task == "key_value":
        return get_key_value_dataloaders_3way(
            num_keys=cfg["kv_num_keys"], vocab_size=cfg["kv_vocab_size"],
            train_frac=cfg["kv_train_frac"], batch_size=cfg["batch_size"], seed=seed,
        )
    return get_modular_addition_dataloaders_3way(
        p=cfg["mod_p"], train_frac=cfg["mod_train_frac"],
        batch_size=cfg["batch_size"], seed=seed,
    )


def _run_component(task, layer_idx, component, cfg, device):
    rows = []
    overrides = ({"kv_num_keys": cfg["kv_num_keys"], "kv_vocab_size": cfg["kv_vocab_size"]}
                 if task == "key_value" else {"mod_p": cfg["mod_p"]})
    for seed in SEEDS:
        ckpt = os.path.join(BASE_DIR, f"seed_{seed}", f"standard_{task}",
                            "checkpoints", "model_best_val.pt")
        if not os.path.exists(ckpt):
            print(f"   [SKIP seed {seed}]"); continue
        model = create_standard_transformer(task, cfg_overrides=overrides, seed=seed)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model = model.to(device).eval()
        _, _, test_loader, _ = _build_loaders(task, cfg, seed)
        base_acc = _eval_accuracy(model, test_loader, task, device)
        print(f"   seed={seed} base test_acc={base_acc:.4f}")
        for r in ABLATION_STRENGTHS:
            scale = 1.0 - r
            if r == 0.0:
                acc = base_acc
            else:
                hook_name, hook_fn = _scale_hook(layer_idx, component, scale)
                acc = _eval_accuracy(model, test_loader, task, device,
                                     fwd_hooks=[(hook_name, hook_fn)])
            rows.append({"task": task, "seed": seed, "layer": layer_idx,
                         "component": component, "r": r, "scale": scale,
                         "test_acc": acc, "delta_test_acc": base_acc - acc,
                         "base_test_acc": base_acc})
    return rows


def _load_existing_summary():
    """Return list of summary dicts already computed in graded_ablation_core."""
    out = []
    if not os.path.exists(EXISTING_CORE):
        return out
    with open(EXISTING_CORE) as f:
        for row in csv.DictReader(f):
            out.append({
                "task": row["task"], "layer": int(row["layer"]),
                "component": row["component"], "r": float(row["ablation_strength_r"]),
                "mean_acc": float(row["mean_ablated_test_acc"]),
                "std_acc": float(row["std_ablated_test_acc"]),
                "n_seeds": int(row["n_seeds"]),
            })
    return out


def _load_existing_l0_attn_kv():
    """Return per-r summary dicts from the earlier KV L0_attn graded run."""
    if not os.path.exists(EXISTING_L0_ATTN_KV):
        return []
    per_r: dict[float, list[float]] = {}
    with open(EXISTING_L0_ATTN_KV) as f:
        for row in csv.DictReader(f):
            r = float(row["r"])
            per_r.setdefault(r, []).append(float(row["test_acc"]))
    return [{"task": "key_value", "layer": 0, "component": "attn",
             "r": r, "mean_acc": float(np.mean(accs)),
             "std_acc": float(np.std(accs)), "n_seeds": len(accs)}
            for r, accs in sorted(per_r.items())]


def _summarize(rows):
    summary = []
    keys = sorted(set((r["task"], r["layer"], r["component"], r["r"]) for r in rows))
    for task, layer, comp, r in keys:
        accs = [row["test_acc"] for row in rows
                if (row["task"], row["layer"], row["component"], row["r"]) == (task, layer, comp, r)]
        summary.append({"task": task, "layer": layer, "component": comp, "r": r,
                        "mean_acc": float(np.mean(accs)),
                        "std_acc": float(np.std(accs)), "n_seeds": len(accs)})
    return summary


def _plot_task(task, summary, out_path):
    color_map = {(0, "attn"): "#2196F3", (0, "mlp"): "#FF9800",
                 (1, "attn"): "#1565C0", (1, "mlp"): "#E65100"}
    label_map = {(0, "attn"): "L0 attention", (0, "mlp"): "L0 MLP",
                 (1, "attn"): "L1 attention", (1, "mlp"): "L1 MLP"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for key, color in color_map.items():
        layer, comp = key
        rows = sorted([s for s in summary if s["task"] == task and s["layer"] == layer
                       and s["component"] == comp], key=lambda x: x["r"])
        if not rows:
            continue
        rs = [s["r"] for s in rows]
        means = [s["mean_acc"] for s in rows]
        stds  = [s["std_acc"]  for s in rows]
        ax.plot(rs, means, "o-", color=color, linewidth=2, markersize=5,
                label=label_map[key])
        ax.fill_between(rs, [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)], color=color, alpha=0.15)
    chance = CHANCE[task]
    ax.axhline(chance, color="red", linestyle=":", linewidth=1, label=f"chance ({chance:.4f})")
    ax.set_xlabel("Ablation strength r  (output scaled by 1 − r)")
    ax.set_ylabel("Test accuracy")
    title = ("Partial ablation dose-response — modular addition (standard, 3 seeds)"
             if task == "modular_addition"
             else "Partial ablation dose-response — key-value retrieval (standard, 3 seeds)")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(PROJECT_ROOT, "configs", "default.yaml")) as f:
        cfg = yaml.safe_load(f)

    existing = _load_existing_summary() + _load_existing_l0_attn_kv()
    have = {(s["task"], s["layer"], s["component"]) for s in existing}
    missing = [c for c in ALL_COMPONENTS if c not in have]
    print(f"Have {len(have)} components, missing {len(missing)}: {missing}")

    new_rows = []
    for task, layer, component in missing:
        print(f"\n=== {task} / layer {layer} / {component} ===")
        new_rows.extend(_run_component(task, layer, component, cfg, device))

    if new_rows:
        with open(os.path.join(OUT_DIR, "missing_runs.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(new_rows[0].keys()))
            writer.writeheader(); writer.writerows(new_rows)
        print(f"\nSaved missing-runs CSV ({len(new_rows)} rows)")
    new_summary = _summarize(new_rows) if new_rows else []
    full_summary = existing + new_summary

    fieldnames = ["task", "layer", "component", "r", "mean_acc", "std_acc", "n_seeds"]
    with open(os.path.join(OUT_DIR, "all_components_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in sorted(full_summary, key=lambda x: (x["task"], x["layer"], x["component"], x["r"])):
            writer.writerow({k: s[k] for k in fieldnames})
    print("\nFull summary written.")

    _plot_task("modular_addition", full_summary,
               os.path.join(OUT_DIR, "modular_addition_partial_ablation.png"))
    _plot_task("key_value", full_summary,
               os.path.join(OUT_DIR, "key_value_partial_ablation.png"))


if __name__ == "__main__":
    main()
