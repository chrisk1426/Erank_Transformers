"""
scripts/compute_erank.py

Thin CLI wrapper around the Week 2 analysis pipeline.

Delegates to run_week2_analysis.main() with argparse support for
--checkpoint_dir, --n_examples, and --device.

Usage:
    python scripts/compute_erank.py
    python scripts/compute_erank.py --n_examples 200 --device cpu
    python scripts/compute_erank.py --checkpoint_dir /path/to/checkpoints
"""

from __future__ import annotations

import argparse
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
from scripts.run_week2_analysis import print_summary_table, print_hypothesis_assessment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute eRank profiles for all trained erank-interp models."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints"),
        help="Directory containing canonical .pt checkpoints.",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=500,
        help="Number of test examples to use for activation extraction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--skip_robustness",
        action="store_true",
        help="Skip the N-stability robustness check.",
    )
    args = parser.parse_args()

    results_dir = os.path.join(PROJECT_ROOT, "results")
    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Device          : {args.device}")
    print(f"Checkpoint dir  : {args.checkpoint_dir}")
    print(f"n_examples      : {args.n_examples}")

    run_verification(figures_dir)

    profiles = compute_all_profiles(
        checkpoint_dir=args.checkpoint_dir,
        n_examples=args.n_examples,
        device=args.device,
    )

    validate_attn_only(
        checkpoint_dir=args.checkpoint_dir,
        results_dir=results_dir,
        n_examples=args.n_examples,
        device=args.device,
    )

    generate_all_plots(profiles, save_dir=figures_dir)

    if not args.skip_robustness:
        run_robustness_check(args.checkpoint_dir, figures_dir, device=args.device)

    print_summary_table(profiles)
    print_hypothesis_assessment(profiles)


if __name__ == "__main__":
    main()
