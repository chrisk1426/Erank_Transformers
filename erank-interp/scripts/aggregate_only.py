"""
scripts/aggregate_only.py

Read all per-run summary.json files from outputs/thesis_additions/
and produce aggregate CSVs, plots, and the final report.

Run this after all parallel experiments finish.
"""

import glob
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import yaml
from train_thesis_additions import aggregate_results, OUTPUT_ROOT


def main():
    import yaml
    with open(os.path.join(PROJECT_ROOT, "configs", "default.yaml")) as f:
        cfg = yaml.safe_load(f)

    pattern = os.path.join(OUTPUT_ROOT, "seed_*", "*", "checkpoints", "summary.json")
    summary_files = sorted(glob.glob(pattern))

    if not summary_files:
        print(f"No summary files found matching: {pattern}")
        sys.exit(1)

    all_summaries = []
    for path in summary_files:
        with open(path) as f:
            all_summaries.append(json.load(f))
        print(f"  Loaded: {path}")

    print(f"\nFound {len(all_summaries)} completed runs. Aggregating...")
    aggregate_results(all_summaries, cfg)
    print("Done.")


if __name__ == "__main__":
    main()
