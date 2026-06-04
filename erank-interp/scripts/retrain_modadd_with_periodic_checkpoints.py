"""
scripts/retrain_modadd_with_periodic_checkpoints.py

Retrain standard modular addition (seed 0) with periodic checkpoint saving
so that the geometry analysis can compute clustering/spectrum metrics densely
over training, not just at best_val/final.

Saves:
  - random_init (epoch 0, before any training)
  - every 250 epochs
  - milestone checkpoints: first train_acc>=0.99, first val_acc>=0.01/0.05/0.10/0.25/0.50/0.75/0.95
  - best_val (updated whenever val_acc improves)
  - final (at early stopping or end of training)
"""

from __future__ import annotations

import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models.transformer import create_standard_transformer
from data.modular_addition import get_modular_addition_dataloaders

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 0
P = 113
TRAIN_FRAC = 0.3
BATCH_SIZE = 256
LR = 3e-4
WEIGHT_DECAY = 1.0
NUM_EPOCHS = 15000  # generous upper bound; early stopping will kick in

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "modadd_periodic_checkpoints", f"seed_{SEED}")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

# Milestone thresholds for one-shot saves
TRAIN_ACC_MILESTONES = {0.99: "first_train_acc_99"}
VAL_ACC_MILESTONES = {
    0.01: "first_val_acc_01",
    0.05: "first_val_acc_05",
    0.10: "first_val_acc_10",
    0.25: "first_val_acc_25",
    0.50: "first_val_acc_50",
    0.75: "first_val_acc_75",
    0.95: "first_val_acc_95",
}

PERIODIC_INTERVAL = 250  # save every N epochs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Seed: {SEED}")
    print(f"Output: {OUT_DIR}")
    print(f"Periodic interval: every {PERIODIC_INTERVAL} epochs")
    print(f"Train acc milestones: {list(TRAIN_ACC_MILESTONES.keys())}")
    print(f"Val acc milestones: {list(VAL_ACC_MILESTONES.keys())}")

    os.makedirs(CKPT_DIR, exist_ok=True)

    # Build model and data
    model = create_standard_transformer(
        "modular_addition", cfg_overrides={"mod_p": P}, seed=SEED,
    )
    train_loader, test_loader = get_modular_addition_dataloaders(
        p=P, train_frac=TRAIN_FRAC, batch_size=BATCH_SIZE, seed=SEED,
    )
    val_loader = test_loader  # consistent with original thesis_additions runs

    _device = torch.device(device)
    model = model.to(_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    # Import helpers
    from training.train_v2 import _run_epoch, _extract_erank_inline

    history = {
        "epoch": [], "train_loss": [], "val_loss": [], "test_loss": [],
        "train_acc": [], "val_acc": [], "test_acc": [],
        "lr": LR, "weight_decay": WEIGHT_DECAY, "task_name": "modular_addition",
        "best_val_acc": 0.0, "best_val_loss": float("inf"),
        "best_val_epoch": None, "sustained_val_epoch": None,
        "erank_time_series": [],
    }

    consecutive_high_val = 0
    best_val_acc = 0.0
    best_val_loss = float("inf")

    # Track which milestones have already fired
    train_acc_fired = {t: False for t in TRAIN_ACC_MILESTONES}
    val_acc_fired = {t: False for t in VAL_ACC_MILESTONES}

    def save_checkpoint(path, epoch, label=None):
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "checkpoint_label": label,
            "train_history": history,
            "model_type": "standard",
            "task_name": "modular_addition",
            "cfg": {
                "n_layers": 2, "d_model": 128, "n_heads": 4, "d_mlp": 512,
                "act_fn": "gelu", "mod_p": P, "mod_train_frac": TRAIN_FRAC,
                "batch_size": BATCH_SIZE, "learning_rate": LR,
                "weight_decay": WEIGHT_DECAY, "seed": SEED,
                "kv_num_keys": 5, "kv_vocab_size": 30, "kv_train_frac": 0.5,
                "hybrid_p": 113, "hybrid_num_kv_pairs": 4,
                "hybrid_num_examples": None, "hybrid_train_frac": 0.5,
                "num_epochs": NUM_EPOCHS, "checkpoint_every": None,
                "n_examples_erank": 500, "val_frac_of_train": None,
            },
        }
        torch.save(state, path)

    def save_and_log(epoch, label):
        """Save a named checkpoint and compute inline eRank."""
        ckpt_path = os.path.join(CKPT_DIR, f"model_{label}.pt")
        save_checkpoint(ckpt_path, epoch, label=label)
        print(f"  [Checkpoint] {label} @ epoch {epoch}")

        # Also save as epoch-numbered for easy iteration
        epoch_path = os.path.join(CKPT_DIR, f"model_epoch_{epoch:06d}.pt")
        if not os.path.exists(epoch_path):
            save_checkpoint(epoch_path, epoch, label=label)

        # Inline eRank
        try:
            profile = _extract_erank_inline(
                model, val_loader, "modular_addition",
                n_examples=500, device=_device,
            )
            entry = {
                "epoch": epoch,
                "train_acc": history["train_acc"][-1] if history["train_acc"] else 0,
                "val_acc": history["val_acc"][-1] if history["val_acc"] else 0,
                "test_acc": history["test_acc"][-1] if history["test_acc"] else 0,
            }
            for l in range(2):
                lk = f"layer_{l}"
                entry[f"layer_{l}_pre"] = profile["erank"][lk]["pre"]
                entry[f"layer_{l}_mid"] = profile["erank"][lk]["mid"]
                entry[f"layer_{l}_post"] = profile["erank"][lk]["post"]
                entry[f"layer_{l}_delta_attn"] = profile["delta_erank_attn"][lk]
                entry[f"layer_{l}_delta_mlp"] = profile["delta_erank_mlp"][lk]
            history["erank_time_series"].append(entry)
            print(
                f"  [eRank @ {epoch}] "
                + "  ".join(
                    f"L{l}: post={entry[f'layer_{l}_post']:.2f}"
                    for l in range(2)
                )
            )
        except Exception as exc:
            print(f"  [eRank @ {epoch}] WARNING: failed — {exc}")

    # ---- Save random_init (epoch 0, before any training) ----
    save_and_log(0, "random_init")

    from tqdm import tqdm

    for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Training modadd (periodic ckpts)"):
        train_loss, train_acc = _run_epoch(
            model, train_loader, "modular_addition", criterion, optimizer, _device,
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, "modular_addition", criterion, None, _device,
        )
        test_loss, test_acc = _run_epoch(
            model, test_loader, "modular_addition", criterion, None, _device,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch:6d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                f"train_acc {train_acc:.4f} | val_acc {val_acc:.4f} | test_acc {test_acc:.4f}"
            )

        # ---- Best-val checkpoint ----
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            history["best_val_acc"] = best_val_acc
            history["best_val_loss"] = best_val_loss
            history["best_val_epoch"] = epoch
            save_checkpoint(os.path.join(CKPT_DIR, "model_best_val.pt"), epoch, label="best_val")

        # ---- Sustained-val tracking ----
        if val_acc > 0.98:
            consecutive_high_val += 1
        else:
            consecutive_high_val = 0

        if consecutive_high_val >= 500 and history["sustained_val_epoch"] is None:
            history["sustained_val_epoch"] = epoch
            print(f"\nSustained val_acc > 98% for 500 epochs: first epoch = {epoch - 499}")

        should_stop = consecutive_high_val >= 500

        # ---- Milestone checkpoints (fire once each) ----
        for threshold, label in TRAIN_ACC_MILESTONES.items():
            if not train_acc_fired[threshold] and train_acc >= threshold:
                train_acc_fired[threshold] = True
                save_and_log(epoch, label)

        for threshold, label in VAL_ACC_MILESTONES.items():
            if not val_acc_fired[threshold] and val_acc >= threshold:
                val_acc_fired[threshold] = True
                save_and_log(epoch, label)

        # ---- Periodic checkpoint every 250 epochs ----
        if epoch % PERIODIC_INTERVAL == 0:
            save_and_log(epoch, f"epoch_{epoch:06d}")

        # ---- Early-stop checkpoint + eRank ----
        if should_stop:
            save_and_log(epoch, f"epoch_{epoch:06d}")
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    # ---- Save final ----
    save_and_log(history["epoch"][-1], "final")

    print(f"\nDone. Final epoch: {history['epoch'][-1]}")
    print(f"Best val epoch: {history['best_val_epoch']}, acc: {history['best_val_acc']:.4f}")
    print(f"Checkpoints saved in: {CKPT_DIR}")
    n_ckpts = len([f for f in os.listdir(CKPT_DIR) if f.endswith(".pt")])
    print(f"Total checkpoints: {n_ckpts}")

    # Print milestone summary
    print("\n--- Milestone summary ---")
    for threshold, label in TRAIN_ACC_MILESTONES.items():
        status = "FIRED" if train_acc_fired[threshold] else "NOT REACHED"
        print(f"  {label}: {status}")
    for threshold, label in VAL_ACC_MILESTONES.items():
        status = "FIRED" if val_acc_fired[threshold] else "NOT REACHED"
        print(f"  {label}: {status}")


if __name__ == "__main__":
    main()
