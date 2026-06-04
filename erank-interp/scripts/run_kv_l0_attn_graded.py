"""
scripts/run_kv_l0_attn_graded.py

Graded ablation of key_value Layer-0 attention (the component left out of
run_graded_ablation_core.py).  Mirrors that script's methodology so the
resulting curves are directly comparable.

Outputs:
  outputs/graded_ablation_l0_attn/kv_l0_attn_graded.csv
  outputs/graded_ablation_l0_attn/kv_l0_attn_dose_response.png
  outputs/graded_ablation_l0_attn/kv_all_four_components_dose_response.png
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

from data.splits import get_key_value_dataloaders_3way
from models.transformer import create_standard_transformer

BASE_DIR  = os.path.join(PROJECT_ROOT, "outputs", "thesis_additions")
OUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "graded_ablation_l0_attn")
SEEDS     = [0, 1, 2]
ABLATION_STRENGTHS = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
CHANCE = 1.0 / 30  # d_vocab_out for key_value


def _eval_accuracy(model, loader, device, fwd_hooks=None):
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
            pred = logits[:, -1, :].argmax(dim=-1)
            n_correct += (pred == labels).sum().item()
            n_total += labels.size(0)
    return n_correct / n_total if n_total > 0 else 0.0


def _make_scale_hook(scale: float):
    name = "blocks.0.hook_attn_out"

    def hook_fn(value, hook):
        return value * scale

    return name, hook_fn


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(PROJECT_ROOT, "configs", "default.yaml")) as f:
        cfg = yaml.safe_load(f)
    print(f"Device: {device}")
    print(f"Strengths: {ABLATION_STRENGTHS}")

    overrides = {"kv_num_keys": cfg["kv_num_keys"],
                 "kv_vocab_size": cfg["kv_vocab_size"]}

    rows = []
    for seed in SEEDS:
        ckpt_path = os.path.join(
            BASE_DIR, f"seed_{seed}", "standard_key_value",
            "checkpoints", "model_best_val.pt"
        )
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {ckpt_path}")
            continue

        print(f"\n  seed={seed}")
        model = create_standard_transformer("key_value", cfg_overrides=overrides, seed=seed)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device).eval()

        _, _, test_loader, _ = get_key_value_dataloaders_3way(
            num_keys=cfg["kv_num_keys"],
            vocab_size=cfg["kv_vocab_size"],
            train_frac=cfg["kv_train_frac"],
            batch_size=cfg["batch_size"],
            seed=seed,
        )

        base_acc = _eval_accuracy(model, test_loader, device)
        print(f"    base test_acc={base_acc:.4f}")

        for r in ABLATION_STRENGTHS:
            scale = 1.0 - r
            if r == 0.0:
                acc = base_acc
            else:
                hook_name, hook_fn = _make_scale_hook(scale)
                acc = _eval_accuracy(model, test_loader, device,
                                     fwd_hooks=[(hook_name, hook_fn)])
            delta = base_acc - acc
            print(f"    r={r:.2f}  scale={scale:.2f}  acc={acc:.4f}  Δ={delta:+.4f}")
            rows.append({"seed": seed, "r": r, "scale": scale,
                         "test_acc": acc, "delta_test_acc": delta,
                         "base_test_acc": base_acc})

    # CSV
    csv_path = os.path.join(OUT_DIR, "kv_l0_attn_graded.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "r", "scale", "test_acc",
                                          "delta_test_acc", "base_test_acc"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # Summary
    print("\nSummary (mean ± std across seeds):")
    print(f"  {'r':>5}  {'scale':>6}  {'mean acc':>10}  {'std':>8}  {'mean Δ':>9}")
    summary = []
    for r in ABLATION_STRENGTHS:
        accs = [row["test_acc"] for row in rows if row["r"] == r]
        deltas = [row["delta_test_acc"] for row in rows if row["r"] == r]
        m, s = float(np.mean(accs)), float(np.std(accs))
        dm = float(np.mean(deltas))
        print(f"  {r:>5.2f}  {1-r:>6.2f}  {m:>10.4f}  {s:>8.4f}  {dm:>+9.4f}")
        summary.append({"r": r, "mean_acc": m, "std_acc": s, "mean_delta": dm})

    # Dose-response plot — KV L0 attn only
    fig, ax = plt.subplots(figsize=(7, 5))
    seed_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    for seed in SEEDS:
        seed_rows = sorted([row for row in rows if row["seed"] == seed],
                           key=lambda x: x["r"])
        if not seed_rows:
            continue
        rs = [row["r"] for row in seed_rows]
        accs = [row["test_acc"] for row in seed_rows]
        ax.plot(rs, accs, "o-", color=seed_colors[seed], alpha=0.5,
                markersize=4, label=f"seed {seed}")
    rs = [s["r"] for s in summary]
    means = [s["mean_acc"] for s in summary]
    stds = [s["std_acc"] for s in summary]
    ax.plot(rs, means, "s-", color="black", linewidth=2, markersize=6,
            label="mean", zorder=5)
    ax.fill_between(rs, [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)], color="gray", alpha=0.2)
    ax.axhline(CHANCE, color="red", linestyle=":", linewidth=1,
               label=f"chance ({CHANCE:.4f})")
    ax.set_xlabel("Ablation strength r")
    ax.set_ylabel("Test accuracy")
    ax.set_title("key_value — L0 attention\nGraded ablation dose-response")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "kv_l0_attn_dose_response.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {p}")

    # Combined plot with the three existing KV curves from graded_ablation_core
    other_csv = os.path.join(PROJECT_ROOT, "outputs", "graded_ablation_core",
                             "graded_ablation_summary.csv")
    if not os.path.exists(other_csv):
        print(f"  [SKIP combined plot — {other_csv} not found]")
        return

    other = {}  # (layer, component) -> list of summary dicts
    with open(other_csv) as f:
        for row in csv.DictReader(f):
            if row["task"] != "key_value":
                continue
            key = (int(row["layer"]), row["component"])
            other.setdefault(key, []).append({
                "r": float(row["ablation_strength_r"]),
                "mean_acc": float(row["mean_ablated_test_acc"]),
                "std_acc": float(row["std_ablated_test_acc"]),
            })

    # Build 2x2: L0 attn (new), L0 MLP, L1 attn, L1 MLP
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    panels = [
        ("L0 attention (new)", summary, "#d62728"),
        ("L0 MLP",  sorted(other.get((0, "mlp"), []),  key=lambda x: x["r"]), "#1f77b4"),
        ("L1 attention", sorted(other.get((1, "attn"), []), key=lambda x: x["r"]), "#2ca02c"),
        ("L1 MLP",  sorted(other.get((1, "mlp"), []),  key=lambda x: x["r"]), "#9467bd"),
    ]
    for ax, (title, data, color) in zip(axes.flatten(), panels):
        if not data:
            ax.set_title(f"{title} (no data)")
            continue
        rs    = [d["r"] for d in data]
        means = [d["mean_acc"] for d in data]
        stds  = [d["std_acc"] for d in data]
        ax.plot(rs, means, "s-", color=color, linewidth=2, markersize=6)
        ax.fill_between(rs, [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)], color=color, alpha=0.18)
        ax.axhline(CHANCE, color="red", linestyle=":", linewidth=1)
        ax.set_xlabel("Ablation strength r")
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"key_value — {title}")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Graded ablation dose-response — all four KV sublayers (mean ± std, n=3 seeds)",
                 fontsize=13)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "kv_all_four_components_dose_response.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {p}")


if __name__ == "__main__":
    main()
