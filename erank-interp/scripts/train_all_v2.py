"""
Retrain all four models with proper train/val/test splits and validation-based
checkpoint selection. Outputs to outputs/control_rerun/.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from data.splits import get_modular_addition_dataloaders_3way, get_key_value_dataloaders_3way
from models.transformer import create_standard_transformer, create_attention_only_transformer
from training.train_v2 import train_v2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "control_rerun")
CKPT_DIR    = os.path.join(OUTPUT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")
SPLITS_DIR  = os.path.join(OUTPUT_ROOT, "splits")
AUDIT_DIR   = os.path.join(OUTPUT_ROOT, "audit")
LOGS_DIR    = os.path.join(OUTPUT_ROOT, "logs")

VAL_FRAC_OF_TRAIN = 0.2

ERANK_EPOCHS_MOD_ADD = [
    1, 5, 10, 25, 50, 100, 200, 300, 500, 750, 1000, 1250, 1500,
    1750, 2000, 2250, 2500, 2750, 3000, 3500, 4000, 5000, 6000,
    7000, 8000, 10000, 15000, 20000, 25000, 30000, 40000, 50000,
]

ERANK_EPOCHS_KV = [1, 5, 10, 25, 50, 100, 200, 500]

EXPERIMENTS = [
    ("standard",       "modular_addition"),
    ("standard",       "key_value"),
    ("attention_only", "modular_addition"),
    ("attention_only", "key_value"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with open(os.path.join(PROJECT_ROOT, "configs", "default.yaml")) as f:
        return yaml.safe_load(f)


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _build_dataloaders(task: str, cfg: dict) -> tuple:
    """Return (train_loader, val_loader, test_loader, split_meta)."""
    if task == "modular_addition":
        return get_modular_addition_dataloaders_3way(
            p=cfg["mod_p"],
            train_frac=cfg["mod_train_frac"],
            val_frac_of_train=VAL_FRAC_OF_TRAIN,
            batch_size=cfg["batch_size"],
            seed=cfg["seed"],
        )
    elif task == "key_value":
        return get_key_value_dataloaders_3way(
            num_keys=cfg["kv_num_keys"],
            vocab_size=cfg["kv_vocab_size"],
            train_frac=cfg["kv_train_frac"],
            val_frac_of_train=VAL_FRAC_OF_TRAIN,
            batch_size=cfg["batch_size"],
            seed=cfg["seed"],
        )
    else:
        raise ValueError(f"Unknown task: {task!r}")


def _build_model(model_type: str, task: str, cfg: dict):
    if task == "modular_addition":
        overrides = {"mod_p": cfg["mod_p"]}
    else:
        overrides = {"kv_num_keys": cfg["kv_num_keys"], "kv_vocab_size": cfg["kv_vocab_size"]}
    if model_type == "standard":
        return create_standard_transformer(task, cfg_overrides=overrides, seed=cfg["seed"])
    else:
        return create_attention_only_transformer(task, cfg_overrides=overrides, seed=cfg["seed"])


def _plot_curves(history: dict, title: str, save_path: str) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = history["epoch"]

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"],   label="val")
    ax_loss.plot(epochs, history["test_loss"],  label="test", linestyle="--", alpha=0.5)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title(f"{title} — Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"],   label="val")
    ax_acc.plot(epochs, history["test_acc"],  label="test", linestyle="--", alpha=0.5)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_title(f"{title} — Accuracy")
    ax_acc.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _write_audit_note() -> None:
    os.makedirs(AUDIT_DIR, exist_ok=True)
    path = os.path.join(AUDIT_DIR, "checkpoint_selection_audit.md")
    content = """\
# Checkpoint Selection Audit

## Findings from the pilot run

### Does the code have a true validation split?
No. The original `data/modular_addition.py` and `data/key_value.py` produce
only a train/test split.  There was no held-out validation set.

### Was validation already logged but not used?
No. The training loop (`training/train.py`) logs and acts on **test accuracy**
only.  The early-stopping rule is `test_acc > 0.98 for 500 consecutive epochs`.
No validation metrics were ever computed or saved.

### Are checkpoints saved often enough to re-select without retraining?
No. Checkpoints were saved every 1000 epochs.  Even if we had retroactively
computed val metrics at those coarse intervals, we would have missed the true
peak (e.g., attention_only_modular_addition peaked near epoch 6383 and no
checkpoint was saved at that epoch without retraining).

### Are historical validation metrics available?
No. Validation metrics do not exist for any of the controlled-run checkpoints
because the validation split did not exist during training.

## Conclusion

Retraining is required for all four runs.

## Fix implemented

- `data/splits.py` carves a validation set from the original training split
  using `seed+1` as the sub-split seed, so the test set remains identical to
  the pilot run.
- `training/train_v2.py` uses `val_acc` (not `test_acc`) for early stopping
  and best-checkpoint selection.  Test metrics are logged as diagnostics only.
- `val_frac_of_train = 0.2` (i.e., 20 % of the original training set becomes
  the validation set, preserving 80 % for training).
"""
    with open(path, "w") as f:
        f.write(content)
    print(f"Audit note saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = _load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for d in [CKPT_DIR, RESULTS_DIR, SPLITS_DIR, AUDIT_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)

    _write_audit_note()

    all_summary_rows: list[dict] = []

    for model_type, task_name in EXPERIMENTS:
        run_name = f"{model_type}_{task_name}"
        print(f"\n{'='*60}")
        print(f"  Run: {run_name}")
        print(f"{'='*60}")

        _seed_all(cfg["seed"])

        train_loader, val_loader, test_loader, split_meta = _build_dataloaders(task_name, cfg)

        # Save split metadata.
        split_path = os.path.join(SPLITS_DIR, f"{run_name}_split_metadata.json")
        with open(split_path, "w") as f:
            json.dump(split_meta, f, indent=2)
        print(f"  Split metadata saved to {split_path}")

        model = _build_model(model_type, task_name, cfg)

        erank_epochs = (
            ERANK_EPOCHS_MOD_ADD if task_name == "modular_addition" else ERANK_EPOCHS_KV
        )
        n_val = split_meta["n_val"]
        erank_n = min(500, n_val)

        history = train_v2(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=cfg["num_epochs"],
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            checkpoint_dir=CKPT_DIR,
            checkpoint_name=run_name,
            device=device,
            task_name=task_name,
            erank_eval_epochs=erank_epochs,
            erank_n_examples=erank_n,
        )

        # Augment checkpoints with cfg / model metadata.
        best_val_path  = os.path.join(CKPT_DIR, f"{run_name}_best_val.pt")
        final_path     = os.path.join(CKPT_DIR, f"{run_name}_final.pt")
        extra_fields = {
            "cfg": {**cfg, "val_frac_of_train": VAL_FRAC_OF_TRAIN},
            "model_type": model_type,
            "task_name": task_name,
        }
        for ckpt_path in [best_val_path, final_path]:
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu")
                ckpt.update(extra_fields)
                torch.save(ckpt, ckpt_path)

        # Test accuracy at best-val epoch.
        best_val_epoch = history["best_val_epoch"]
        if best_val_epoch is not None and best_val_epoch <= len(history["epoch"]):
            idx = history["epoch"].index(best_val_epoch)
            test_acc_at_best_val = history["test_acc"][idx]
        else:
            test_acc_at_best_val = None

        final_epoch     = history["epoch"][-1]
        final_train_acc = history["train_acc"][-1]
        final_val_acc   = history["val_acc"][-1]
        final_test_acc  = history["test_acc"][-1]

        # Checkpoint summary JSON.
        summary = {
            "run": run_name,
            "task": task_name,
            "architecture": model_type,
            "seed": cfg["seed"],
            "lr": cfg["learning_rate"],
            "weight_decay": cfg["weight_decay"],
            "max_epochs": cfg["num_epochs"],
            "val_frac_of_train": VAL_FRAC_OF_TRAIN,
            "best_val_epoch": best_val_epoch,
            "best_val_acc": history["best_val_acc"],
            "best_val_loss": history["best_val_loss"],
            "sustained_val_epoch": history["sustained_val_epoch"],
            "final_epoch": final_epoch,
            "final_train_acc": final_train_acc,
            "final_val_acc": final_val_acc,
            "final_test_acc": final_test_acc,
            "test_acc_at_best_val": test_acc_at_best_val,
            "selected_checkpoint": "best_val",
            "selected_checkpoint_path": best_val_path,
        }
        summary_path = os.path.join(CKPT_DIR, f"{run_name}_checkpoint_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Checkpoint summary saved to {summary_path}")

        # Training curves JSON.
        curves_path = os.path.join(RESULTS_DIR, f"{run_name}_curves.json")
        with open(curves_path, "w") as f:
            json.dump(history, f)
        print(f"  Training curves saved to {curves_path}")

        # Training curves plot.
        fig_path = os.path.join(RESULTS_DIR, f"{run_name}_training.png")
        _plot_curves(history, title=run_name, save_path=fig_path)
        print(f"  Training plot saved to {fig_path}")

        all_summary_rows.append({
            "run": run_name,
            "architecture": model_type,
            "task": task_name,
            "best_val_epoch": best_val_epoch,
            "best_val_acc": history["best_val_acc"],
            "sustained_val_epoch": history["sustained_val_epoch"],
            "final_epoch": final_epoch,
            "final_test_acc": final_test_acc,
            "test_acc_at_best_val": test_acc_at_best_val,
        })

    # Aggregate CSV.
    csv_path = os.path.join(CKPT_DIR, "all_runs_summary.csv")
    fieldnames = [
        "run", "architecture", "task",
        "best_val_epoch", "best_val_acc", "sustained_val_epoch",
        "final_epoch", "final_test_acc", "test_acc_at_best_val",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_summary_rows)
    print(f"\nAll-runs summary saved to {csv_path}")


if __name__ == "__main__":
    main()
