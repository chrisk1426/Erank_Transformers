"""
analysis/erank_robustness.py

Robustness check: how stable are eRank estimates as a function of N_examples?

Recomputes eRank for the standard modular_addition model at N = 100, 250,
500, 750, 1000 and plots eRank vs N for each (layer, hook_point).
Flat lines indicate stable estimates; large variation signals unreliability.

Output: figures/erank_n_stability.png
"""

from __future__ import annotations

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

from analysis.erank import compute_erank
from analysis.extract_activations import load_model_and_extract


N_VALUES = [100, 250, 500, 750, 1000]
# Colors per hook point type
COLORS = {
    "pre":  "#4C72B0",
    "mid":  "#DD8452",
    "post": "#55A868",
}
LINESTYLES = ["-", "--"]  # layer 0, layer 1


def run_robustness_check(
    checkpoint_dir: str,
    figures_dir: str,
    device: str = "cuda",
) -> None:
    """
    Extract activations at various N and plot eRank stability.

    Args:
        checkpoint_dir: Directory with canonical checkpoints.
        figures_dir: Directory to save the figure.
        device: "cuda" or "cpu".
    """
    ckpt_path = os.path.join(checkpoint_dir, "standard_modular_addition.pt")
    n_max = max(N_VALUES)

    print(f"Extracting {n_max} activations for standard_modular_addition...")
    acts_full = load_model_and_extract(
        ckpt_path,
        task_name="modular_addition",
        model_type="standard",
        n_examples=n_max,
        device=device,
    )

    # Determine n_layers from the keys present.
    n_layers = sum(1 for k in acts_full if k.endswith("hook_resid_pre"))

    # For each N, slice the activation matrices and compute eRank.
    hook_points = ["pre", "mid", "post"]
    # results[layer][hook_point] = list of eRank values (one per N in N_VALUES)
    results: dict[int, dict[str, list[float]]] = {
        layer: {hp: [] for hp in hook_points}
        for layer in range(n_layers)
    }

    for n in N_VALUES:
        for layer in range(n_layers):
            for hp in hook_points:
                hook_name = f"blocks.{layer}.hook_resid_{hp}"
                mat = acts_full[hook_name][:n]  # (n, d_model)
                er = compute_erank(mat)
                results[layer][hp].append(er)

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4), sharey=False)
    if n_layers == 1:
        axes = [axes]

    for layer, ax in enumerate(axes):
        for hp in hook_points:
            ls = LINESTYLES[layer % len(LINESTYLES)]
            ax.plot(
                N_VALUES,
                results[layer][hp],
                color=COLORS[hp],
                linestyle=ls,
                marker="o",
                label=f"resid_{hp}",
            )
        ax.set_xlabel("N examples")
        ax.set_ylabel("eRank")
        ax.set_title(f"Layer {layer} — eRank stability")
        ax.legend()

    fig.suptitle("eRank vs N_examples (standard / modular_addition)", fontsize=11)
    fig.tight_layout()

    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, "erank_n_stability.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    fig_dir  = os.path.join(PROJECT_ROOT, "figures")
    run_robustness_check(ckpt_dir, fig_dir, device=device)
