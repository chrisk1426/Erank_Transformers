"""
scripts/run_week2_analysis.py

End-to-end Week 2 analysis pipeline for erank-interp.

Steps:
  1. Verify eRank on synthetic matrices
  2. Compute eRank profiles for all 4 trained models
  3. Run attention-only control validation
  4. Generate all plots
  5. Run eRank robustness check (N stability)
  6. Print summary table and initial hypothesis assessment

Usage:
    python scripts/run_week2_analysis.py [--n_examples N] [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from analysis.compute_erank_profiles import compute_all_profiles
from analysis.erank_robustness import run_robustness_check
from analysis.plot_erank import generate_all_plots
from scripts.validate_attn_only import validate_attn_only
from scripts.verify_erank import run_verification


def print_summary_table(profiles: dict) -> None:
    """Print the eRank profile summary table."""
    header = (
        f"{'Model':<35} {'Layer':>5}  {'pre':>6}  {'mid':>6}  {'post':>6}"
        f"  {'Δ_attn':>7}  {'Δ_mlp':>7}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("eRank PROFILE SUMMARY")
    print(sep)
    print(header)
    print("-" * len(header))

    for run_name, profile in profiles.items():
        n_layers = len(profile["erank"])
        for layer in range(n_layers):
            key = f"layer_{layer}"
            e  = profile["erank"][key]
            da = profile["delta_erank_attn"][key]
            dm = profile["delta_erank_mlp"][key]
            print(
                f"{run_name:<35} {layer:>5}  "
                f"{e['pre']:>6.3f}  {e['mid']:>6.3f}  {e['post']:>6.3f}"
                f"  {da:>+7.3f}  {dm:>+7.3f}"
            )

    print(sep)


def print_hypothesis_assessment(profiles: dict) -> None:
    """Print an initial assessment of H1, H2, and the control check."""
    print("\n" + "=" * 55)
    print("HYPOTHESIS ASSESSMENT")
    print("=" * 55)

    tasks = ["modular_addition", "key_value"]
    for task in tasks:
        std_key = f"standard_{task}"
        if std_key not in profiles:
            continue

        profile = profiles[std_key]
        n_layers = len(profile["erank"])

        # Sum delta eRank across layers.
        total_da = sum(
            profile["delta_erank_attn"][f"layer_{l}"] for l in range(n_layers)
        )
        total_dm = sum(
            profile["delta_erank_mlp"][f"layer_{l}"] for l in range(n_layers)
        )
        print(f"\n  Task: {task}")
        print(f"    Total Δ_attn = {total_da:+.3f}")
        print(f"    Total Δ_mlp  = {total_dm:+.3f}")

        if task == "key_value":
            supported = total_da > total_dm
            print(
                f"    H1 (attention dominates on key_value): "
                f"{'SUPPORTED' if supported else 'NOT SUPPORTED'}"
            )
        elif task == "modular_addition":
            supported = total_dm > total_da
            print(
                f"    H2 (MLP dominates on modular_addition): "
                f"{'SUPPORTED' if supported else 'NOT SUPPORTED'}"
            )

    # Control check: attention-only models should have Δ_mlp ≈ 0.
    print("\n  Control (attention-only models, Δ_mlp ≈ 0):")
    for run_name, profile in profiles.items():
        if not run_name.startswith("attention_only_"):
            continue
        n_layers = len(profile["erank"])
        max_dm = max(
            abs(profile["delta_erank_mlp"][f"layer_{l}"]) for l in range(n_layers)
        )
        ok = max_dm < 0.1
        print(
            f"    {run_name}: max |Δ_mlp| = {max_dm:.4f}  "
            f"[{'OK' if ok else 'NOT OK'}]"
        )

    print("=" * 55)


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 2 eRank analysis pipeline")
    parser.add_argument("--n_examples", type=int, default=500)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--skip_robustness", action="store_true",
        help="Skip the N-stability robustness check (faster run)."
    )
    args = parser.parse_args()

    ckpt_dir    = os.path.join(PROJECT_ROOT, "checkpoints")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    figures_dir = os.path.join(PROJECT_ROOT, "figures")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"n_examples: {args.n_examples}")

    # ---------------------------------------------------------------
    # Step 1: Verify eRank on synthetic data
    # ---------------------------------------------------------------
    print("\n" + "=" * 55)
    print("STEP 1: eRank verification on synthetic matrices")
    print("=" * 55)
    run_verification(figures_dir)

    # ---------------------------------------------------------------
    # Step 2: Compute eRank profiles for all 4 models
    # ---------------------------------------------------------------
    print("\n" + "=" * 55)
    print("STEP 2: Computing eRank profiles")
    print("=" * 55)
    profiles = compute_all_profiles(
        checkpoint_dir=ckpt_dir,
        n_examples=args.n_examples,
        device=args.device,
    )

    # ---------------------------------------------------------------
    # Step 3: Attention-only control validation
    # ---------------------------------------------------------------
    print("\n" + "=" * 55)
    print("STEP 3: Attention-only control validation")
    print("=" * 55)
    validate_attn_only(
        checkpoint_dir=ckpt_dir,
        results_dir=results_dir,
        n_examples=args.n_examples,
        device=args.device,
    )

    # ---------------------------------------------------------------
    # Step 4: Generate all plots
    # ---------------------------------------------------------------
    print("\n" + "=" * 55)
    print("STEP 4: Generating plots")
    print("=" * 55)
    generate_all_plots(profiles, save_dir=figures_dir)

    # ---------------------------------------------------------------
    # Step 5: eRank robustness check
    # ---------------------------------------------------------------
    if not args.skip_robustness:
        print("\n" + "=" * 55)
        print("STEP 5: eRank N-stability robustness check")
        print("=" * 55)
        run_robustness_check(ckpt_dir, figures_dir, device=args.device)

    # ---------------------------------------------------------------
    # Step 6: Summary table + hypothesis assessment
    # ---------------------------------------------------------------
    print_summary_table(profiles)
    print_hypothesis_assessment(profiles)

    print(f"\nAll Week 2 outputs saved.")
    print(f"  Profiles : {os.path.join(results_dir, 'erank_profiles.json')}")
    print(f"  Figures  : {figures_dir}/")


if __name__ == "__main__":
    main()
