"""
scripts/train_all.py

Train all four model/task combinations for erank-interp and summarise results.

Models trained:
  1. standard       × modular_addition  (predict MLP dominates eRank growth)
  2. standard       × key_value         (predict attention dominates eRank growth)
  3. attention_only × modular_addition  (control: MLP eRank contribution should vanish)
  4. attention_only × key_value         (control: validate attention-alone routing)

Outputs per run:
  - checkpoints/{model_type}_{task_name}.pt          final model checkpoint
  - results/{model_type}_{task_name}_curves.json     training history
  - figures/{model_type}_{task_name}_training.png    loss and accuracy curves
"""

from __future__ import annotations

import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Allow imports from the project root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from data.key_value import get_key_value_dataloaders
from data.modular_addition import get_modular_addition_dataloaders
from models.transformer import create_attention_only_transformer, create_standard_transformer
from training.train import train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _build_dataloaders(task: str, cfg: dict):
    """Return (train_loader, test_loader) for the given task."""
    if task == "modular_addition":
        return get_modular_addition_dataloaders(
            p=cfg["mod_p"],
            train_frac=cfg["mod_train_frac"],
            batch_size=cfg["batch_size"],
            seed=cfg["seed"],
        )
    elif task == "key_value":
        return get_key_value_dataloaders(
            num_keys=cfg["kv_num_keys"],
            vocab_size=cfg["kv_vocab_size"],
            train_frac=cfg["kv_train_frac"],
            batch_size=cfg["batch_size"],
            seed=cfg["seed"],
        )
    else:
        raise ValueError(f"Unknown task: {task!r}")


def _build_model(model_type: str, task: str, cfg: dict):
    """Return an untrained HookedTransformer."""
    if task == "modular_addition":
        cfg_overrides = {"mod_p": cfg["mod_p"]}
    else:
        cfg_overrides = {"kv_num_keys": cfg["kv_num_keys"], "kv_vocab_size": cfg["kv_vocab_size"]}
    if model_type == "standard":
        return create_standard_transformer(task, cfg_overrides=cfg_overrides, seed=cfg["seed"])
    elif model_type == "attention_only":
        return create_attention_only_transformer(task, cfg_overrides=cfg_overrides, seed=cfg["seed"])
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")


def _plot_curves(history: dict, title: str, save_path: str) -> None:
    """Save a 2-panel figure (loss | accuracy) for the training history."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = history["epoch"]

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["test_loss"], label="test")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title(f"{title} — Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["test_acc"], label="test")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_title(f"{title} — Accuracy")
    ax_acc.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Figure saved to {save_path}")


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

# (model_type, task_name, weight_decay)
# All models use the same weight_decay (1.0, canonical for grokking per Nanda et al.)
# and the same early stopping rule (test_acc > 98% for 500 consecutive epochs).
EXPERIMENTS: list[tuple[str, str, float]] = [
    ("standard",       "modular_addition", 1.0),
    ("standard",       "key_value",        1.0),
    ("attention_only", "modular_addition", 1.0),
    ("attention_only", "key_value",        1.0),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", type=int, default=None,
        help="Run only experiment index 0-3 (default: run all sequentially)."
    )
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
    cfg = _load_config(config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: {config_path}\n")

    os.makedirs(os.path.join(PROJECT_ROOT, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "figures"), exist_ok=True)

    experiments = EXPERIMENTS if args.run is None else [EXPERIMENTS[args.run]]

    summary: list[dict] = []

    for model_type, task_name, weight_decay in experiments:
        run_name = f"{model_type}_{task_name}"
        max_epochs = cfg["num_epochs"]
        print(f"\n{'='*60}")
        print(f"  Run: {run_name}")
        print(f"  weight_decay={weight_decay}  max_epochs={max_epochs}")
        print(f"{'='*60}")

        _seed_all(cfg["seed"])
        train_loader, test_loader = _build_dataloaders(task_name, cfg)
        model = _build_model(model_type, task_name, cfg)

        history = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=max_epochs,
            lr=cfg["learning_rate"],
            weight_decay=weight_decay,
            checkpoint_dir=os.path.join(PROJECT_ROOT, "checkpoints"),
            checkpoint_every=cfg["checkpoint_every"],
            device=device,
            task_name=task_name,
        )

        # Save the final model under a canonical name for compute_erank.py.
        final_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", f"{run_name}.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": history["epoch"][-1],
                "train_history": history,
                "model_type": model_type,
                "task_name": task_name,
                "cfg": cfg,
                "weight_decay": weight_decay,
                "lr": cfg["learning_rate"],
                "max_epochs": max_epochs,
            },
            final_ckpt,
        )
        print(f"  Canonical checkpoint saved to {final_ckpt}")

        # Save training curves as JSON.
        curves_path = os.path.join(PROJECT_ROOT, "results", f"{run_name}_curves.json")
        with open(curves_path, "w") as f:
            json.dump(history, f)
        print(f"  Curves saved to {curves_path}")

        # Plot and save training curves.
        fig_path = os.path.join(PROJECT_ROOT, "figures", f"{run_name}_training.png")
        _plot_curves(history, title=run_name, save_path=fig_path)

        final_test_acc = history["test_acc"][-1]
        final_train_acc = history["train_acc"][-1]
        summary.append(
            {
                "run": run_name,
                "epochs_trained": history["epoch"][-1],
                "final_train_acc": final_train_acc,
                "final_test_acc": final_test_acc,
            }
        )

    # Print summary table.
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    header = f"{'Run':<35} {'Epochs':>8} {'Train Acc':>10} {'Test Acc':>10}"
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['run']:<35} {row['epochs_trained']:>8d} "
            f"{row['final_train_acc']:>10.4f} {row['final_test_acc']:>10.4f}"
        )


if __name__ == "__main__":
    main()
