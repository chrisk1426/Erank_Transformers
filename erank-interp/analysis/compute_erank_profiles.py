"""
analysis/compute_erank_profiles.py

eRank profile computation for erank-interp.

Takes extracted activation matrices and computes eRank at every
(layer, hook_point) combination, including delta eRank values that
attribute representational complexity growth to attention vs MLP.
"""

from __future__ import annotations

import json
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from analysis.erank import compute_erank
from analysis.extract_activations import load_model_and_extract


# ---------------------------------------------------------------------------
# Profile computation
# ---------------------------------------------------------------------------

def compute_erank_profile(
    activations: dict[str, torch.Tensor],
    n_layers: int,
) -> dict:
    """
    Compute eRank at every (layer, hook_point) from extracted activations.

    Args:
        activations: Dict mapping hook names to activation matrices
                     of shape (n_examples, d_model), as returned by
                     extract_activations().
        n_layers: Number of transformer layers.

    Returns:
        Dictionary with structure:
        {
            "erank": {
                "layer_0": {"pre": float, "mid": float, "post": float},
                "layer_1": {"pre": float, "mid": float, "post": float},
                ...
            },
            "delta_erank_attn": {"layer_0": float, "layer_1": float, ...},
            "delta_erank_mlp":  {"layer_0": float, "layer_1": float, ...},
        }
    """
    erank_vals: dict[str, dict[str, float]] = {}
    delta_attn: dict[str, float] = {}
    delta_mlp: dict[str, float] = {}

    for layer in range(n_layers):
        key = f"layer_{layer}"
        pre  = compute_erank(activations[f"blocks.{layer}.hook_resid_pre"])
        mid  = compute_erank(activations[f"blocks.{layer}.hook_resid_mid"])
        post = compute_erank(activations[f"blocks.{layer}.hook_resid_post"])

        erank_vals[key] = {"pre": pre, "mid": mid, "post": post}
        delta_attn[key] = mid - pre
        delta_mlp[key]  = post - mid

    return {
        "erank": erank_vals,
        "delta_erank_attn": delta_attn,
        "delta_erank_mlp": delta_mlp,
    }


# ---------------------------------------------------------------------------
# All-models pipeline
# ---------------------------------------------------------------------------

RUN_NAMES: list[tuple[str, str]] = [
    ("standard",       "modular_addition"),
    ("standard",       "key_value"),
    ("attention_only", "modular_addition"),
    ("attention_only", "key_value"),
]


def compute_all_profiles(
    checkpoint_dir: str,
    n_examples: int = 500,
    device: str = "cuda",
) -> dict[str, dict]:
    """
    Compute eRank profiles for all 4 trained models.

    Args:
        checkpoint_dir: Directory containing the canonical .pt checkpoints.
        n_examples: Number of test examples to use per model.
        device: "cuda" or "cpu".

    Returns:
        Dictionary mapping run names to eRank profiles:
        {
            "standard_modular_addition":       {profile},
            "standard_key_value":              {profile},
            "attention_only_modular_addition": {profile},
            "attention_only_key_value":        {profile},
        }
        Also saves results to PROJECT_ROOT/results/erank_profiles.json.
    """
    import torch as _torch

    # n_layers is fixed (2) for all models in this project.
    # We read it from one checkpoint to be safe.
    n_layers = 2  # matches configs/default.yaml

    all_profiles: dict[str, dict] = {}

    for model_type, task_name in RUN_NAMES:
        run_name = f"{model_type}_{task_name}"
        ckpt_path = os.path.join(checkpoint_dir, f"{run_name}.pt")

        print(f"\n{'='*50}")
        print(f"  Computing eRank profile: {run_name}")
        print(f"{'='*50}")

        activations = load_model_and_extract(
            ckpt_path,
            task_name=task_name,
            model_type=model_type,
            n_examples=n_examples,
            device=device,
        )

        profile = compute_erank_profile(activations, n_layers=n_layers)
        all_profiles[run_name] = profile

        # Print per-layer summary.
        for layer in range(n_layers):
            key = f"layer_{layer}"
            e = profile["erank"][key]
            da = profile["delta_erank_attn"][key]
            dm = profile["delta_erank_mlp"][key]
            print(
                f"  Layer {layer}: pre={e['pre']:.3f}  mid={e['mid']:.3f}  "
                f"post={e['post']:.3f}  Δ_attn={da:+.3f}  Δ_mlp={dm:+.3f}"
            )

    # Save to results/
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "erank_profiles.json")
    with open(save_path, "w") as f:
        json.dump(all_profiles, f, indent=2)
    print(f"\nSaved eRank profiles to {save_path}")

    return all_profiles


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    profiles = compute_all_profiles(ckpt_dir, n_examples=500, device=device)
    print("\nDone.")
