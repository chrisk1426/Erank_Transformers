"""
Step 3: For attention_only_modular_addition, compare peak vs final checkpoint.
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
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models.transformer import create_attention_only_transformer
from analysis.extract_activations import extract_activations
from analysis.compute_erank_profiles import compute_erank_profile
from data.splits import get_modular_addition_dataloaders_3way

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "control_rerun")
CKPT_DIR    = os.path.join(OUTPUT_ROOT, "checkpoints")
OUT_DIR     = os.path.join(OUTPUT_ROOT, "attn_only_mod_add")
PLOTS_DIR   = os.path.join(OUT_DIR, "plots")

N_EXAMPLES  = 1000
TASK_NAME   = "modular_addition"
RUN_NAME    = "attention_only_modular_addition"

COLOR_PRE  = "#4C72B0"
COLOR_MID  = "#DD8452"
COLOR_POST = "#55A868"
DPI = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rebuild_model(ckpt: dict):
    cfg  = ckpt["cfg"]
    seed = cfg.get("seed", 42)
    overrides = {"mod_p": cfg["mod_p"]}
    model = create_attention_only_transformer(TASK_NAME, cfg_overrides=overrides, seed=seed)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def _rebuild_val_loader(cfg: dict):
    seed = cfg.get("seed", 42)
    vfot = cfg.get("val_frac_of_train", 0.2)
    bs   = cfg.get("batch_size", 256)
    _, val_loader, _, _ = get_modular_addition_dataloaders_3way(
        p=cfg["mod_p"],
        train_frac=cfg["mod_train_frac"],
        val_frac_of_train=vfot,
        batch_size=bs,
        seed=seed,
    )
    return val_loader


def _rebuild_all_loaders(cfg: dict):
    seed = cfg.get("seed", 42)
    vfot = cfg.get("val_frac_of_train", 0.2)
    bs   = cfg.get("batch_size", 256)
    return get_modular_addition_dataloaders_3way(
        p=cfg["mod_p"],
        train_frac=cfg["mod_train_frac"],
        val_frac_of_train=vfot,
        batch_size=bs,
        seed=seed,
    )


def _get_pred_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, 2, :]


def _evaluate(model, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            preds  = _get_pred_logits(logits)
            loss   = criterion(preds, labels)
            total_loss += loss.item()
            n_correct  += (preds.argmax(dim=-1) == labels).sum().item()
            n_total    += labels.size(0)
    return total_loss / len(loader), n_correct / n_total


def _profile_summary(profile: dict, n_layers: int) -> dict:
    row = {}
    for l in range(n_layers):
        lk = f"layer_{l}"
        row[f"layer_{l}_pre"]        = profile["erank"][lk]["pre"]
        row[f"layer_{l}_mid"]        = profile["erank"][lk]["mid"]
        row[f"layer_{l}_post"]       = profile["erank"][lk]["post"]
        row[f"layer_{l}_delta_attn"] = profile["delta_erank_attn"][lk]
        row[f"layer_{l}_delta_mlp"]  = profile["delta_erank_mlp"][lk]
    return row


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_side_by_side(peak_profile: dict, final_profile: dict, n_layers: int, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    layers = list(range(n_layers))

    def _arr(p, hook):
        return [p["erank"][f"layer_{l}"][hook] for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, profile, title in zip(axes, [peak_profile, final_profile], ["Peak (best_val)", "Final"]):
        ax.plot(layers, _arr(profile, "pre"),  marker="o", color=COLOR_PRE,  label="resid_pre")
        ax.plot(layers, _arr(profile, "mid"),  marker="s", color=COLOR_MID,  label="resid_mid")
        ax.plot(layers, _arr(profile, "post"), marker="^", color=COLOR_POST, label="resid_post")
        ax.set_xlabel("Layer")
        ax.set_ylabel("eRank")
        ax.set_title(f"attn-only modular_addition\n{title}")
        ax.set_xticks(layers)
        ax.legend()
    fig.tight_layout()
    path = os.path.join(save_dir, "peak_vs_final_erank.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_delta_comparison(peak_profile: dict, final_profile: dict, n_layers: int, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    layers = list(range(n_layers))
    x = np.arange(n_layers)
    w = 0.35

    peak_da  = [peak_profile["delta_erank_attn"][f"layer_{l}"]  for l in layers]
    final_da = [final_profile["delta_erank_attn"][f"layer_{l}"] for l in layers]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, peak_da,  w, label="peak Δ_attn",  color="#4C72B0", alpha=0.8)
    ax.bar(x + w / 2, final_da, w, label="final Δ_attn", color="#4C72B0", alpha=0.4)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Δ_attn (eRank)")
    ax.set_title("Attention Delta: Peak vs Final")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend()
    fig.tight_layout()
    path = os.path.join(save_dir, "peak_vs_final_delta.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Report template
# ---------------------------------------------------------------------------

def _write_report(row_peak: dict, row_final: dict, save_path: str) -> None:
    def _fmt(row, key):
        val = row.get(key)
        return f"{val:.4f}" if isinstance(val, float) else str(val)

    content = f"""\
# Attention-Only Modular Addition: Peak vs Final Checkpoint

## Overview

This report compares the geometry of the attention-only modular addition model
at two checkpoints:

- **Peak** (best validation accuracy): epoch {row_peak['epoch']}, val_acc={_fmt(row_peak,'val_acc')}
- **Final** (last saved epoch): epoch {row_final['epoch']}, val_acc={_fmt(row_final,'val_acc')}

## Accuracy Comparison

| Checkpoint | Epoch | Train Acc | Val Acc | Test Acc |
|------------|-------|-----------|---------|----------|
| peak       | {row_peak['epoch']} | {_fmt(row_peak,'train_acc')} | {_fmt(row_peak,'val_acc')} | {_fmt(row_peak,'test_acc')} |
| final      | {row_final['epoch']} | {_fmt(row_final,'train_acc')} | {_fmt(row_final,'val_acc')} | {_fmt(row_final,'test_acc')} |

## eRank Profile Comparison

### Layer 0

| Checkpoint | pre | mid | post | Δ_attn | Δ_mlp |
|------------|-----|-----|------|--------|-------|
| peak  | {_fmt(row_peak,'layer_0_pre')} | {_fmt(row_peak,'layer_0_mid')} | {_fmt(row_peak,'layer_0_post')} | {_fmt(row_peak,'layer_0_delta_attn')} | {_fmt(row_peak,'layer_0_delta_mlp')} |
| final | {_fmt(row_final,'layer_0_pre')} | {_fmt(row_final,'layer_0_mid')} | {_fmt(row_final,'layer_0_post')} | {_fmt(row_final,'layer_0_delta_attn')} | {_fmt(row_final,'layer_0_delta_mlp')} |

### Layer 1

| Checkpoint | pre | mid | post | Δ_attn | Δ_mlp |
|------------|-----|-----|------|--------|-------|
| peak  | {_fmt(row_peak,'layer_1_pre')} | {_fmt(row_peak,'layer_1_mid')} | {_fmt(row_peak,'layer_1_post')} | {_fmt(row_peak,'layer_1_delta_attn')} | {_fmt(row_peak,'layer_1_delta_mlp')} |
| final | {_fmt(row_final,'layer_1_pre')} | {_fmt(row_final,'layer_1_mid')} | {_fmt(row_final,'layer_1_post')} | {_fmt(row_final,'layer_1_delta_attn')} | {_fmt(row_final,'layer_1_delta_mlp')} |

## Interpretation

At the **peak checkpoint** (epoch {row_peak['epoch']}, val_acc={_fmt(row_peak,'val_acc')}), the geometry showed:
- Layer-0 Δ_attn = {_fmt(row_peak,'layer_0_delta_attn')}, indicating the attention stage
  {"expanded" if row_peak.get("layer_0_delta_attn", 0) > 0 else "compressed"} the residual stream's effective dimensionality.
- Layer-1 Δ_attn = {_fmt(row_peak,'layer_1_delta_attn')}.
- Δ_mlp = 0 exactly for both layers (by construction: attn-only model).

By the **final checkpoint** (epoch {row_final['epoch']}, val_acc={_fmt(row_final,'val_acc')}), the geometry changed as follows:
- Layer-0 Δ_attn shifted from {_fmt(row_peak,'layer_0_delta_attn')} to {_fmt(row_final,'layer_0_delta_attn')}.
- Layer-1 Δ_attn shifted from {_fmt(row_peak,'layer_1_delta_attn')} to {_fmt(row_final,'layer_1_delta_attn')}.
- Overall pre/mid/post eRank values shifted across layers (see table above).

### Source of Degradation

Based on the eRank profiles above, the degradation in validation accuracy from
the peak to the final checkpoint appears to be associated with changes in the
attention-stage geometry (Δ_attn), since Δ_mlp = 0 by construction.

Whether the dominant mechanism is:
(a) **Spectrum concentration** — if pre/mid/post eRank values at the final
    checkpoint are lower, indicating more low-rank residual streams, or
(b) **Attention-stage instability** — if Δ_attn changes sign or magnitude
    substantially between peak and final, suggesting the attention sublayers
    are reorganising in a way that reduces representational diversity,
can be read directly from the eRank values in the table above.

Note: this report does not claim causality; it describes geometric changes
between the two saved states.
"""
    with open(save_path, "w") as f:
        f.write(content)
    print(f"  Saved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    _device = torch.device(device_str)

    for d in [OUT_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)

    # Load checkpoints.
    peak_path  = os.path.join(CKPT_DIR, f"{RUN_NAME}_best_val.pt")
    final_path = os.path.join(CKPT_DIR, f"{RUN_NAME}_final.pt")

    print(f"Loading peak checkpoint: {peak_path}")
    peak_ckpt  = torch.load(peak_path,  map_location=_device)
    print(f"Loading final checkpoint: {final_path}")
    final_ckpt = torch.load(final_path, map_location=_device)

    peak_model  = _rebuild_model(peak_ckpt).to(_device)
    final_model = _rebuild_model(final_ckpt).to(_device)

    cfg = peak_ckpt["cfg"]
    train_loader, val_loader, test_loader, _ = _rebuild_all_loaders(cfg)

    n_val = len(val_loader.dataset)
    n_use = min(N_EXAMPLES, n_val)

    results = {}
    for label, model, ckpt in [("peak", peak_model, peak_ckpt), ("final", final_model, final_ckpt)]:
        print(f"\n--- {label} (epoch {ckpt['epoch']}) ---")
        model.eval()

        train_loss, train_acc = _evaluate(model, train_loader, _device)
        val_loss,   val_acc   = _evaluate(model, val_loader,   _device)
        test_loss,  test_acc  = _evaluate(model, test_loader,  _device)
        print(f"  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

        acts    = extract_activations(model, val_loader, TASK_NAME, n_examples=n_use, device=device_str)
        profile = compute_erank_profile(acts, n_layers=model.cfg.n_layers)
        n_layers = model.cfg.n_layers

        row = {
            "checkpoint_type": label,
            "epoch": ckpt["epoch"],
            "train_acc": train_acc,
            "val_acc":   val_acc,
            "test_acc":  test_acc,
        }
        row.update(_profile_summary(profile, n_layers))
        results[label] = (row, profile, n_layers)

    row_peak,  profile_peak,  n_layers = results["peak"]
    row_final, profile_final, _        = results["final"]

    # Save CSV.
    csv_path = os.path.join(OUT_DIR, "peak_vs_final_summary.csv")
    fieldnames = ["checkpoint_type", "epoch", "train_acc", "val_acc", "test_acc"]
    for l in range(n_layers):
        fieldnames += [f"layer_{l}_pre", f"layer_{l}_mid", f"layer_{l}_post",
                       f"layer_{l}_delta_attn", f"layer_{l}_delta_mlp"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row_peak)
        writer.writerow(row_final)
    print(f"\nSaved {csv_path}")

    # Save plots.
    _plot_side_by_side(profile_peak, profile_final, n_layers, PLOTS_DIR)
    _plot_delta_comparison(profile_peak, profile_final, n_layers, PLOTS_DIR)

    # Save report.
    report_path = os.path.join(OUT_DIR, "peak_vs_final_report.md")
    _write_report(row_peak, row_final, report_path)

    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
