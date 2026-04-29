"""
training/train_v2.py

Updated training loop for erank-interp with proper validation-based
checkpoint selection.

Key differences from train.py:
- Uses val_acc (not test_acc) for early stopping and best-checkpoint tracking.
- Logs train / val / test metrics every 100 epochs.
- Saves {checkpoint_name}_best_val.pt when val_acc improves.
- Optionally computes inline eRank profiles at specified epochs.
- Returns a rich history dict including erank_time_series.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Prediction-position helper (duplicated from train.py to avoid cross-imports)
# ---------------------------------------------------------------------------

def _get_logits_at_pred_position(logits: torch.Tensor, task_name: str) -> torch.Tensor:
    if task_name == "modular_addition":
        return logits[:, 2, :]
    elif task_name == "key_value":
        return logits[:, -1, :]
    else:
        raise ValueError(f"Unknown task_name: {task_name!r}")


# ---------------------------------------------------------------------------
# Single-pass epoch runner
# ---------------------------------------------------------------------------

def _run_epoch(
    model: HookedTransformer,
    loader: DataLoader,
    task_name: str,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    n_correct = 0
    n_examples = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            pred_logits = _get_logits_at_pred_position(logits, task_name)
            loss = criterion(pred_logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_correct += (pred_logits.argmax(dim=-1) == labels).sum().item()
            n_examples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = n_correct / n_examples
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Inline eRank extraction (no disk I/O — model already in memory)
# ---------------------------------------------------------------------------

def _extract_erank_inline(
    model: HookedTransformer,
    val_loader: DataLoader,
    task_name: str,
    n_examples: int,
    device: torch.device,
) -> dict:
    """Extract activations from val set and return eRank profile."""
    from analysis.extract_activations import extract_activations
    from analysis.compute_erank_profiles import compute_erank_profile

    model.eval()
    acts = extract_activations(model, val_loader, task_name, n_examples=n_examples, device=str(device))
    return compute_erank_profile(acts, n_layers=model.cfg.n_layers)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_v2(
    model: HookedTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    lr: float = 3e-4,
    weight_decay: float = 1.0,
    checkpoint_dir: str = "checkpoints",
    checkpoint_name: str = "model",
    device: str = "cuda",
    task_name: str = "modular_addition",
    erank_eval_epochs: list[int] | None = None,
    erank_n_examples: int = 500,
) -> dict:
    """
    Train a HookedTransformer with validation-based checkpoint selection.

    Early stopping triggers when val_acc > 0.98 for 500 consecutive epochs.
    The best-val checkpoint is saved whenever val_acc strictly improves.
    Test metrics are logged for diagnostics only and never used for selection.

    Args:
        model: Untrained HookedTransformer.
        train_loader: DataLoader for training examples.
        val_loader: DataLoader for validation examples.
        test_loader: DataLoader for test examples.
        num_epochs: Maximum number of training epochs.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        checkpoint_dir: Directory for checkpoint files.
        checkpoint_name: Base name for checkpoint files.
        device: "cuda" or "cpu".
        task_name: "modular_addition" or "key_value".
        erank_eval_epochs: Epoch numbers at which to compute inline eRank.
        erank_n_examples: Number of val examples used for inline eRank.

    Returns:
        History dict with per-epoch metrics and erank_time_series.
    """
    _device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(checkpoint_dir, exist_ok=True)

    erank_eval_set = set(erank_eval_epochs) if erank_eval_epochs else set()

    history: dict = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
        "lr": lr,
        "weight_decay": weight_decay,
        "task_name": task_name,
        "best_val_acc": 0.0,
        "best_val_loss": float("inf"),
        "best_val_epoch": None,
        "sustained_val_epoch": None,
        "erank_time_series": [],
    }

    consecutive_high_val = 0
    best_val_acc = 0.0
    best_val_loss = float("inf")

    best_val_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_best_val.pt")

    def _save_checkpoint(path: str, epoch: int, extra: dict | None = None):
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_history": history,
            "model_type": getattr(model.cfg, "_model_type", None),
            "task_name": task_name,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)

    for epoch in tqdm(range(1, num_epochs + 1), desc=f"Training {task_name}"):
        train_loss, train_acc = _run_epoch(
            model, train_loader, task_name, criterion, optimizer, _device
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, task_name, criterion, None, _device
        )
        test_loss, test_acc = _run_epoch(
            model, test_loader, task_name, criterion, None, _device
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
                f"Epoch {epoch:6d} | "
                f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                f"train_acc {train_acc:.4f} | val_acc {val_acc:.4f} | "
                f"test_acc {test_acc:.4f}"
            )

        # Best-val checkpoint (tie-break: lower val_loss, then earlier epoch).
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            history["best_val_acc"] = best_val_acc
            history["best_val_loss"] = best_val_loss
            history["best_val_epoch"] = epoch
            _save_checkpoint(best_val_path, epoch)

        # Sustained-val tracking.
        if val_acc > 0.98:
            consecutive_high_val += 1
        else:
            consecutive_high_val = 0

        if consecutive_high_val >= 500 and history["sustained_val_epoch"] is None:
            history["sustained_val_epoch"] = epoch
            print(f"\nSustained val_acc > 98% for 500 epochs: first epoch = {epoch - 499}")

        should_stop = consecutive_high_val >= 500

        # Inline eRank — evaluated BEFORE the early-stop break so the
        # stopping epoch is always included in erank_time_series.
        if epoch in erank_eval_set or should_stop:
            try:
                profile = _extract_erank_inline(
                    model, val_loader, task_name,
                    n_examples=erank_n_examples, device=_device,
                )
                n_layers = model.cfg.n_layers
                entry: dict = {
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                }
                for l in range(n_layers):
                    lk = f"layer_{l}"
                    entry[f"layer_{l}_pre"]        = profile["erank"][lk]["pre"]
                    entry[f"layer_{l}_mid"]        = profile["erank"][lk]["mid"]
                    entry[f"layer_{l}_post"]       = profile["erank"][lk]["post"]
                    entry[f"layer_{l}_delta_attn"] = profile["delta_erank_attn"][lk]
                    entry[f"layer_{l}_delta_mlp"]  = profile["delta_erank_mlp"][lk]
                history["erank_time_series"].append(entry)
                print(
                    f"  [eRank @ {epoch}] "
                    + "  ".join(
                        f"L{l}: pre={entry[f'layer_{l}_pre']:.2f} "
                        f"mid={entry[f'layer_{l}_mid']:.2f} "
                        f"post={entry[f'layer_{l}_post']:.2f}"
                        for l in range(n_layers)
                    )
                )
            except Exception as exc:
                print(f"  [eRank @ {epoch}] WARNING: failed — {exc}")

        # Early stopping.
        if should_stop:
            print(f"\nEarly stopping at epoch {epoch}: val_acc > 98% for 500 consecutive epochs.")
            break

    # Save final checkpoint.
    final_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_final.pt")
    _save_checkpoint(final_path, history["epoch"][-1])
    print(f"Final checkpoint saved to {final_path}")
    print(f"Best-val checkpoint: epoch {history['best_val_epoch']}, val_acc={history['best_val_acc']:.4f}")

    return history
