"""
scripts/run_single_experiment.py

Run a single seed/model_type/task experiment and save all artefacts.
Designed to be called in parallel by run_thesis_parallel.sh.

Usage:
    python scripts/run_single_experiment.py \
        --seed 0 --model_type standard --task key_value
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import yaml
import torch
from train_thesis_additions import run_one, _load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",       type=int,  required=True)
    parser.add_argument("--model_type", type=str,  required=True,
                        choices=["standard", "attention_only"])
    parser.add_argument("--task",       type=str,  required=True,
                        choices=["key_value", "modular_addition", "hybrid_retrieve_add"])
    parser.add_argument("--max_epochs", type=int,  default=None,
                        help="Override max epochs from config (e.g. cap attn-only runs)")
    args = parser.parse_args()

    cfg    = _load_config()
    if args.max_epochs is not None:
        cfg["num_epochs"] = args.max_epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[run_single] seed={args.seed} model={args.model_type} task={args.task} "
          f"max_epochs={cfg['num_epochs']} device={device}")
    summary = run_one(args.model_type, args.task, args.seed, cfg, device)
    print(f"[run_single] DONE seed={args.seed} model={args.model_type} task={args.task} "
          f"test_acc={summary.get('test_acc')} notes={summary.get('notes')}")

if __name__ == "__main__":
    main()
