"""
analysis/plot_erank.py

Plotting functions for eRank profiles in erank-interp.

Generates four types of visualizations:
  1. eRank vs layer (one plot per model)
  2. Delta eRank bar charts (attention vs MLP contribution per task)
  3. Standard vs attention-only overlay (one plot per task)
  4. Control validation plot (attention-only Δ_mlp ≈ 0 check)
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
COLOR_PRE  = "#4C72B0"   # blue
COLOR_MID  = "#DD8452"   # orange
COLOR_POST = "#55A868"   # green
COLOR_ATTN = "#4C72B0"   # blue (for delta bars)
COLOR_MLP  = "#C44E52"   # red  (for delta bars)

DPI = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _n_layers(profile: dict) -> int:
    return len(profile["erank"])


def _layer_indices(profile: dict) -> list[int]:
    return list(range(_n_layers(profile)))


def _erank_arrays(profile: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pre, mid, post) arrays across layers."""
    layers = _layer_indices(profile)
    pre  = np.array([profile["erank"][f"layer_{l}"]["pre"]  for l in layers])
    mid  = np.array([profile["erank"][f"layer_{l}"]["mid"]  for l in layers])
    post = np.array([profile["erank"][f"layer_{l}"]["post"] for l in layers])
    return pre, mid, post


def _delta_arrays(profile: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (delta_attn, delta_mlp) arrays across layers."""
    layers = _layer_indices(profile)
    da = np.array([profile["delta_erank_attn"][f"layer_{l}"] for l in layers])
    dm = np.array([profile["delta_erank_mlp"][f"layer_{l}"]  for l in layers])
    return da, dm


# ---------------------------------------------------------------------------
# Plot type 1: eRank vs layer per model
# ---------------------------------------------------------------------------

def plot_erank_per_model(profiles: dict, save_dir: str = "figures") -> None:
    """Generate one eRank-vs-layer plot per model (4 total)."""
    os.makedirs(save_dir, exist_ok=True)

    for run_name, profile in profiles.items():
        parts = run_name.split("_", 1)
        model_type = parts[0]
        task_name  = parts[1] if len(parts) > 1 else run_name
        # handle "attention_only" prefix
        if run_name.startswith("attention_only_"):
            model_type = "attention_only"
            task_name  = run_name[len("attention_only_"):]

        layers = _layer_indices(profile)
        pre, mid, post = _erank_arrays(profile)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(layers, pre,  marker="o", color=COLOR_PRE,  label="resid_pre")
        ax.plot(layers, mid,  marker="s", color=COLOR_MID,  label="resid_mid")
        ax.plot(layers, post, marker="^", color=COLOR_POST, label="resid_post")

        ax.set_xlabel("Layer")
        ax.set_ylabel("eRank")
        ax.set_title(f"{model_type} × {task_name}")
        ax.set_xticks(layers)
        ax.legend()
        fig.tight_layout()

        fname = f"erank_{run_name}.png"
        fpath = os.path.join(save_dir, fname)
        fig.savefig(fpath, dpi=DPI)
        plt.close(fig)
        print(f"  Saved {fpath}")


# ---------------------------------------------------------------------------
# Plot type 2: Delta eRank bar chart per task (standard models only)
# ---------------------------------------------------------------------------

def plot_delta_erank(profiles: dict, save_dir: str = "figures") -> None:
    """Generate grouped delta-eRank bar charts for standard models (2 plots)."""
    os.makedirs(save_dir, exist_ok=True)

    tasks = ["modular_addition", "key_value"]
    for task_name in tasks:
        run_name = f"standard_{task_name}"
        if run_name not in profiles:
            print(f"  Warning: {run_name} not in profiles, skipping.")
            continue

        profile = profiles[run_name]
        layers = _layer_indices(profile)
        da, dm = _delta_arrays(profile)

        x = np.arange(len(layers))
        width = 0.35

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width / 2, da, width, label="Δ_attn (attention)", color=COLOR_ATTN)
        ax.bar(x + width / 2, dm, width, label="Δ_mlp  (MLP)",       color=COLOR_MLP)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Layer")
        ax.set_ylabel("ΔeRank")
        ax.set_title(f"eRank Growth by Component — {task_name}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in layers])
        ax.legend()
        fig.tight_layout()

        fname = f"delta_erank_{task_name}.png"
        fpath = os.path.join(save_dir, fname)
        fig.savefig(fpath, dpi=DPI)
        plt.close(fig)
        print(f"  Saved {fpath}")


# ---------------------------------------------------------------------------
# Plot type 3: Standard vs attention-only overlay per task
# ---------------------------------------------------------------------------

def plot_standard_vs_attn_only(profiles: dict, save_dir: str = "figures") -> None:
    """Overlay standard and attention-only eRank profiles for each task (2 plots)."""
    os.makedirs(save_dir, exist_ok=True)

    tasks = ["modular_addition", "key_value"]
    for task_name in tasks:
        std_key  = f"standard_{task_name}"
        attn_key = f"attention_only_{task_name}"

        if std_key not in profiles or attn_key not in profiles:
            print(f"  Warning: missing profile for {task_name}, skipping comparison plot.")
            continue

        std_profile  = profiles[std_key]
        attn_profile = profiles[attn_key]
        layers = _layer_indices(std_profile)

        std_pre,  std_mid,  std_post  = _erank_arrays(std_profile)
        attn_pre, attn_mid, attn_post = _erank_arrays(attn_profile)

        fig, ax = plt.subplots(figsize=(7, 4))

        # Standard — solid lines
        ax.plot(layers, std_pre,  color=COLOR_PRE,  linestyle="-",  marker="o", label="std resid_pre")
        ax.plot(layers, std_mid,  color=COLOR_MID,  linestyle="-",  marker="s", label="std resid_mid")
        ax.plot(layers, std_post, color=COLOR_POST, linestyle="-",  marker="^", label="std resid_post")

        # Attention-only — dashed lines
        ax.plot(layers, attn_pre,  color=COLOR_PRE,  linestyle="--", marker="o", label="attn-only resid_pre")
        ax.plot(layers, attn_mid,  color=COLOR_MID,  linestyle="--", marker="s", label="attn-only resid_mid")
        ax.plot(layers, attn_post, color=COLOR_POST, linestyle="--", marker="^", label="attn-only resid_post")

        ax.set_xlabel("Layer")
        ax.set_ylabel("eRank")
        ax.set_title(f"Standard vs Attention-Only — {task_name}")
        ax.set_xticks(layers)
        ax.legend(fontsize=7)
        fig.tight_layout()

        fname = f"erank_comparison_{task_name}.png"
        fpath = os.path.join(save_dir, fname)
        fig.savefig(fpath, dpi=DPI)
        plt.close(fig)
        print(f"  Saved {fpath}")


# ---------------------------------------------------------------------------
# Plot type 4: Control validation (attention-only Δ_mlp ≈ 0)
# ---------------------------------------------------------------------------

def plot_control_validation(profiles: dict, save_dir: str = "figures") -> None:
    """
    For each attention-only model, plot |erank_mid - erank_post| per layer.
    Should be near-zero (sanity check that MLP is truly disabled).
    """
    os.makedirs(save_dir, exist_ok=True)

    attn_runs = [k for k in profiles if k.startswith("attention_only_")]
    if not attn_runs:
        print("  No attention-only profiles found, skipping control validation plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    for run_name in sorted(attn_runs):
        profile = profiles[run_name]
        layers = _layer_indices(profile)
        _, mid, post = _erank_arrays(profile)
        diff = np.abs(mid - post)
        label = run_name.replace("attention_only_", "attn-only ")
        ax.plot(layers, diff, marker="o", label=label)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("|eRank(resid_mid) − eRank(resid_post)|")
    ax.set_title("Control Validation: Attention-Only Δ_mlp ≈ 0")
    ax.set_xticks(list(range(max(len(_layer_indices(profiles[r])) for r in attn_runs))))
    ax.legend()
    fig.tight_layout()

    fname = "control_validation.png"
    fpath = os.path.join(save_dir, fname)
    fig.savefig(fpath, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {fpath}")


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def generate_all_plots(profiles: dict, save_dir: str = "figures") -> None:
    """Generate all four plot types from the eRank profiles."""
    print("\n[Plots] eRank per model...")
    plot_erank_per_model(profiles, save_dir)

    print("[Plots] Delta eRank bar charts...")
    plot_delta_erank(profiles, save_dir)

    print("[Plots] Standard vs attention-only overlays...")
    plot_standard_vs_attn_only(profiles, save_dir)

    print("[Plots] Control validation...")
    plot_control_validation(profiles, save_dir)

    print("[Plots] All plots generated.")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    results_path = os.path.join(PROJECT_ROOT, "results", "erank_profiles.json")
    figures_dir  = os.path.join(PROJECT_ROOT, "figures")

    with open(results_path) as f:
        profiles = json.load(f)

    generate_all_plots(profiles, save_dir=figures_dir)
