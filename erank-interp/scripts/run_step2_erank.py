"""
Step 2: Rerun eRank analysis on validation-selected checkpoints.
Reads from outputs/control_rerun/checkpoints/.
Writes to outputs/control_rerun/erank/.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models.transformer import create_standard_transformer, create_attention_only_transformer
from analysis.extract_activations import extract_activations
from analysis.compute_erank_profiles import compute_erank_profile
from analysis.erank import compute_erank
from data.splits import get_modular_addition_dataloaders_3way, get_key_value_dataloaders_3way

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_ROOT  = os.path.join(PROJECT_ROOT, "outputs", "control_rerun")
CKPT_DIR     = os.path.join(OUTPUT_ROOT, "checkpoints")
ERANK_DIR    = os.path.join(OUTPUT_ROOT, "erank")
TABLES_DIR   = os.path.join(ERANK_DIR, "tables")
SPECTRA_DIR  = os.path.join(ERANK_DIR, "raw_spectra")
META_DIR     = os.path.join(ERANK_DIR, "metadata")
PLOTS_DIR    = os.path.join(ERANK_DIR, "plots")

N_EXAMPLES = 1000

RUNS = [
    ("standard",       "modular_addition"),
    ("standard",       "key_value"),
    ("attention_only", "modular_addition"),
    ("attention_only", "key_value"),
]

# ---------------------------------------------------------------------------
# Colours (match plot_erank.py)
# ---------------------------------------------------------------------------
COLOR_PRE  = "#4C72B0"
COLOR_MID  = "#DD8452"
COLOR_POST = "#55A868"
COLOR_ATTN = "#4C72B0"
COLOR_MLP  = "#C44E52"
DPI = 150


# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------

def _rebuild_model(ckpt: dict):
    cfg      = ckpt["cfg"]
    mtype    = ckpt["model_type"]
    task     = ckpt["task_name"]
    seed     = cfg.get("seed", 42)

    if task == "modular_addition":
        overrides = {"mod_p": cfg["mod_p"]}
    else:
        overrides = {"kv_num_keys": cfg["kv_num_keys"], "kv_vocab_size": cfg["kv_vocab_size"]}

    if mtype == "standard":
        model = create_standard_transformer(task, cfg_overrides=overrides, seed=seed)
    else:
        model = create_attention_only_transformer(task, cfg_overrides=overrides, seed=seed)

    model.load_state_dict(ckpt["model_state_dict"])
    return model


def _rebuild_val_loader(ckpt: dict, n_examples: int):
    cfg  = ckpt["cfg"]
    task = ckpt["task_name"]
    seed = cfg.get("seed", 42)
    vfot = cfg.get("val_frac_of_train", 0.2)
    bs   = cfg.get("batch_size", 256)

    if task == "modular_addition":
        _, val_loader, _, _ = get_modular_addition_dataloaders_3way(
            p=cfg["mod_p"],
            train_frac=cfg["mod_train_frac"],
            val_frac_of_train=vfot,
            batch_size=bs,
            seed=seed,
        )
    else:
        _, val_loader, _, _ = get_key_value_dataloaders_3way(
            num_keys=cfg["kv_num_keys"],
            vocab_size=cfg["kv_vocab_size"],
            train_frac=cfg["kv_train_frac"],
            val_frac_of_train=vfot,
            batch_size=bs,
            seed=seed,
        )
    return val_loader


# ---------------------------------------------------------------------------
# Raw singular-value extraction (for saving spectra)
# ---------------------------------------------------------------------------

def _centered_svd(act: torch.Tensor) -> np.ndarray:
    """Return centered singular values for an activation matrix."""
    A = act.detach().float().cpu().numpy()
    A = A - A.mean(axis=0)
    sigma = np.linalg.svd(A, full_matrices=False, compute_uv=False)
    return sigma


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_erank_per_model(profiles: dict, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for run_name, info in profiles.items():
        profile = info["profile"]
        n_layers = len(profile["erank"])
        layers = list(range(n_layers))
        pre  = [profile["erank"][f"layer_{l}"]["pre"]  for l in layers]
        mid  = [profile["erank"][f"layer_{l}"]["mid"]  for l in layers]
        post = [profile["erank"][f"layer_{l}"]["post"] for l in layers]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(layers, pre,  marker="o", color=COLOR_PRE,  label="resid_pre")
        ax.plot(layers, mid,  marker="s", color=COLOR_MID,  label="resid_mid")
        ax.plot(layers, post, marker="^", color=COLOR_POST, label="resid_post")
        ax.set_xlabel("Layer")
        ax.set_ylabel("eRank")
        ax.set_title(f"{run_name} (best_val)")
        ax.set_xticks(layers)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(save_dir, f"erank_{run_name}_best_val.png")
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(f"  Saved {path}")


def _plot_delta_bar(profiles: dict, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for run_name, info in profiles.items():
        if not run_name.startswith("standard_"):
            continue
        profile = info["profile"]
        n_layers = len(profile["erank"])
        layers = list(range(n_layers))
        da = [profile["delta_erank_attn"][f"layer_{l}"] for l in layers]
        dm = [profile["delta_erank_mlp"][f"layer_{l}"]  for l in layers]

        x = np.arange(n_layers)
        w = 0.35
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - w / 2, da, w, label="Δ_attn", color=COLOR_ATTN)
        ax.bar(x + w / 2, dm, w, label="Δ_mlp",  color=COLOR_MLP)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Layer")
        ax.set_ylabel("ΔeRank")
        ax.set_title(f"eRank Growth by Component — {run_name}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in layers])
        ax.legend()
        fig.tight_layout()
        path = os.path.join(save_dir, f"delta_erank_{run_name}_best_val.png")
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(f"  Saved {path}")


def _plot_std_vs_attn(profiles: dict, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for task in ("modular_addition", "key_value"):
        std_key  = f"standard_{task}"
        attn_key = f"attention_only_{task}"
        if std_key not in profiles or attn_key not in profiles:
            continue
        sp = profiles[std_key]["profile"]
        ap = profiles[attn_key]["profile"]
        n_layers = len(sp["erank"])
        layers = list(range(n_layers))

        def _arr(p, hook):
            return [p["erank"][f"layer_{l}"][hook] for l in layers]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(layers, _arr(sp, "pre"),  color=COLOR_PRE,  linestyle="-",  marker="o", label="std pre")
        ax.plot(layers, _arr(sp, "mid"),  color=COLOR_MID,  linestyle="-",  marker="s", label="std mid")
        ax.plot(layers, _arr(sp, "post"), color=COLOR_POST, linestyle="-",  marker="^", label="std post")
        ax.plot(layers, _arr(ap, "pre"),  color=COLOR_PRE,  linestyle="--", marker="o", label="attn pre")
        ax.plot(layers, _arr(ap, "mid"),  color=COLOR_MID,  linestyle="--", marker="s", label="attn mid")
        ax.plot(layers, _arr(ap, "post"), color=COLOR_POST, linestyle="--", marker="^", label="attn post")
        ax.set_xlabel("Layer")
        ax.set_ylabel("eRank")
        ax.set_title(f"Standard vs Attention-Only — {task}")
        ax.set_xticks(layers)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(save_dir, f"erank_comparison_{task}_best_val.png")
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(f"  Saved {path}")


def _plot_control_validation(profiles: dict, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    attn_runs = [k for k in profiles if k.startswith("attention_only_")]
    if not attn_runs:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for run_name in sorted(attn_runs):
        profile = profiles[run_name]["profile"]
        n_layers = len(profile["erank"])
        layers = list(range(n_layers))
        diff = [abs(profile["erank"][f"layer_{l}"]["mid"] - profile["erank"][f"layer_{l}"]["post"])
                for l in layers]
        ax.plot(layers, diff, marker="o", label=run_name.replace("attention_only_", "attn-only "))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("|eRank(mid) − eRank(post)|")
    ax.set_title("Control Validation: Attention-Only Δ_mlp ≈ 0")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(save_dir, "control_validation_best_val.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_summary_bar(profiles: dict, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    names, sum_da, sum_dm = [], [], []
    for run_name, info in profiles.items():
        profile = info["profile"]
        n_layers = len(profile["erank"])
        da = sum(profile["delta_erank_attn"][f"layer_{l}"] for l in range(n_layers))
        dm = sum(profile["delta_erank_mlp"][f"layer_{l}"]  for l in range(n_layers))
        names.append(run_name)
        sum_da.append(da)
        sum_dm.append(dm)

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, sum_da, w, label="sum Δ_attn", color=COLOR_ATTN)
    ax.bar(x + w / 2, sum_dm, w, label="sum Δ_mlp",  color=COLOR_MLP)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Sum ΔeRank")
    ax.set_title("Total eRank Growth: Attention vs MLP")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(save_dir, "summary_bar_best_val.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for d in [TABLES_DIR, SPECTRA_DIR, META_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)

    profiles: dict[str, dict] = {}
    erank_by_layer_rows: list[dict] = []
    erank_summary_rows: list[dict] = []
    eval_meta: dict[str, dict] = {}

    for model_type, task_name in RUNS:
        run_name = f"{model_type}_{task_name}"
        ckpt_path = os.path.join(CKPT_DIR, f"{run_name}_best_val.pt")
        summary_path = os.path.join(CKPT_DIR, f"{run_name}_checkpoint_summary.json")

        print(f"\n{'='*55}")
        print(f"  {run_name}  (best_val checkpoint)")
        print(f"{'='*55}")

        _device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        ckpt = torch.load(ckpt_path, map_location=_device)
        model = _rebuild_model(ckpt)
        model = model.to(_device)
        model.eval()

        val_loader = _rebuild_val_loader(ckpt, N_EXAMPLES)
        n_val = len(val_loader.dataset)
        n_use = min(N_EXAMPLES, n_val)

        # Extract activations.
        acts = extract_activations(model, val_loader, task_name, n_examples=n_use, device=device)
        profile = compute_erank_profile(acts, n_layers=model.cfg.n_layers)

        epoch = ckpt.get("epoch", "unknown")

        # Load final test acc from summary JSON if available.
        final_val_acc = final_test_acc = None
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summ = json.load(f)
            final_val_acc  = summ.get("best_val_acc")
            final_test_acc = summ.get("test_acc_at_best_val")

        profiles[run_name] = {"profile": profile, "epoch": epoch}

        n_layers = model.cfg.n_layers

        # Save raw spectra and populate table rows.
        sum_da = sum_dm = 0.0
        for l in range(n_layers):
            lk = f"layer_{l}"
            for hook in ("pre", "mid", "post"):
                hook_name = f"blocks.{l}.hook_resid_{hook}"
                sigma = _centered_svd(acts[hook_name])
                spec_path = os.path.join(
                    SPECTRA_DIR, f"{run_name}_best_val_layer{l}_{hook}.npy"
                )
                np.save(spec_path, sigma)

            da = profile["delta_erank_attn"][lk]
            dm = profile["delta_erank_mlp"][lk]
            sum_da += da
            sum_dm += dm

            erank_by_layer_rows.append({
                "run": run_name,
                "checkpoint_type": "best_val",
                "epoch": epoch,
                "layer": l,
                "pre":   profile["erank"][lk]["pre"],
                "mid":   profile["erank"][lk]["mid"],
                "post":  profile["erank"][lk]["post"],
                "delta_attn": da,
                "delta_mlp":  dm,
            })

            print(
                f"  Layer {l}: pre={profile['erank'][lk]['pre']:.3f}  "
                f"mid={profile['erank'][lk]['mid']:.3f}  "
                f"post={profile['erank'][lk]['post']:.3f}  "
                f"Δ_attn={da:+.3f}  Δ_mlp={dm:+.3f}"
            )

        erank_summary_rows.append({
            "run": run_name,
            "checkpoint_type": "best_val",
            "epoch": epoch,
            "sum_delta_attn": sum_da,
            "sum_delta_mlp":  sum_dm,
            "final_val_acc":  final_val_acc,
            "final_test_acc": final_test_acc,
        })

        eval_meta[run_name] = {
            "n_examples_requested": N_EXAMPLES,
            "n_examples_used": n_use,
            "n_val_available": n_val,
            "seed": ckpt["cfg"].get("seed", 42),
            "checkpoint_path": ckpt_path,
            "epoch": epoch,
        }

    # Save tables.
    layer_csv = os.path.join(TABLES_DIR, "erank_by_layer.csv")
    with open(layer_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "checkpoint_type", "epoch", "layer",
                                               "pre", "mid", "post", "delta_attn", "delta_mlp"])
        writer.writeheader()
        writer.writerows(erank_by_layer_rows)
    print(f"\nSaved {layer_csv}")

    summary_csv = os.path.join(TABLES_DIR, "erank_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "checkpoint_type", "epoch",
                                               "sum_delta_attn", "sum_delta_mlp",
                                               "final_val_acc", "final_test_acc"])
        writer.writeheader()
        writer.writerows(erank_summary_rows)
    print(f"Saved {summary_csv}")

    meta_path = os.path.join(META_DIR, "eval_sample_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(eval_meta, f, indent=2)
    print(f"Saved {meta_path}")

    # Generate plots.
    print("\n[Plots]")
    _plot_erank_per_model(profiles, PLOTS_DIR)
    _plot_delta_bar(profiles, PLOTS_DIR)
    _plot_std_vs_attn(profiles, PLOTS_DIR)
    _plot_control_validation(profiles, PLOTS_DIR)
    _plot_summary_bar(profiles, PLOTS_DIR)

    print("\nStep 2 complete.")


if __name__ == "__main__":
    main()
