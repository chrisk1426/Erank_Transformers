"""
training/train.py

Generic training loop for erank-interp.

Trains a HookedTransformer on either the modular addition or key-value
retrieval task. Logs metrics, saves checkpoints, and supports early stopping
once the model achieves high test accuracy.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer


def _get_logits_at_pred_position(
    logits: torch.Tensor,
    task_name: str,
) -> torch.Tensor:
    """
    Extract per-example prediction logits at the relevant sequence position.

    - modular_addition: position 2 (the '=' token, i.e. the last input token)
    - key_value:        position -1 (the query key, last token in sequence)

    Args:
        logits: Tensor of shape (batch, n_ctx, d_vocab_out).
        task_name: "modular_addition" or "key_value".

    Returns:
        Tensor of shape (batch, d_vocab_out).
    """
    if task_name == "modular_addition":
        return logits[:, 2, :]
    elif task_name == "key_value":
        return logits[:, -1, :]
    else:
        raise ValueError(f"Unknown task_name: {task_name!r}. Expected 'modular_addition' or 'key_value'.")


def _run_epoch(
    model: HookedTransformer,
    loader: DataLoader,
    task_name: str,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one full pass over a DataLoader.

    When optimizer is provided the model is put in training mode and weights
    are updated. When optimizer is None the model is put in eval mode and
    no gradients are computed.

    Args:
        model: The HookedTransformer to run.
        loader: DataLoader yielding (inputs, labels) batches.
        task_name: "modular_addition" or "key_value".
        criterion: CrossEntropyLoss instance.
        optimizer: AdamW optimizer for training, or None for evaluation.
        device: Device to move tensors to.

    Returns:
        (avg_loss, accuracy) over the full DataLoader.
    """
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

            logits = model(inputs)                                   # (batch, n_ctx, d_vocab_out)
            pred_logits = _get_logits_at_pred_position(logits, task_name)  # (batch, d_vocab_out)

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


def train(
    model: HookedTransformer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    checkpoint_dir: str = "checkpoints",
    checkpoint_every: int = 1000,
    device: str = "cuda",
    task_name: str = "modular_addition",
) -> dict:
    """
    Train a HookedTransformer and return the full training history.

    Logs train/test loss and accuracy every 100 epochs. Saves checkpoints
    every checkpoint_every epochs and a final checkpoint after training ends.
    Applies early stopping once test accuracy exceeds 98% for 500 consecutive
    evaluations.

    Args:
        model: Untrained HookedTransformer.
        train_loader: DataLoader for training examples.
        test_loader: DataLoader for test examples.
        num_epochs: Maximum number of training epochs.
        lr: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW (use 1.0 for modular addition
            to encourage grokking; 0.01 for key-value).
        checkpoint_dir: Directory to save checkpoints. Created if absent.
        checkpoint_every: Save a checkpoint every this many epochs.
        device: "cuda" or "cpu". The model is moved to this device.
        task_name: "modular_addition" or "key_value". Controls which sequence
            position is used for the prediction logits.

    Returns:
        History dict with keys "epoch", "train_loss", "test_loss",
        "train_acc", "test_acc" — each a list of values recorded per epoch.
    """
    _device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(checkpoint_dir, exist_ok=True)

    history: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
        "lr": lr,
        "weight_decay": weight_decay,
        "task_name": task_name,
    }

    consecutive_high_acc = 0

    for epoch in tqdm(range(1, num_epochs + 1), desc=f"Training {task_name}"):
        train_loss, train_acc = _run_epoch(
            model, train_loader, task_name, criterion, optimizer, _device
        )
        test_loss, test_acc = _run_epoch(
            model, test_loader, task_name, criterion, None, _device
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        # Logging every 100 epochs.
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch:6d} | "
                f"train_loss {train_loss:.4f} | test_loss {test_loss:.4f} | "
                f"train_acc {train_acc:.4f} | test_acc {test_acc:.4f}"
            )

        # Periodic checkpoint.
        if epoch % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"{task_name}_epoch_{epoch}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_history": history,
                },
                ckpt_path,
            )

        # Early stopping: test_acc > 98% for 500 consecutive evaluations.
        if test_acc > 0.98:
            consecutive_high_acc += 1
        else:
            consecutive_high_acc = 0

        if consecutive_high_acc >= 500:
            print(f"\nEarly stopping at epoch {epoch}: test_acc > 98% for 500 consecutive epochs.")
            break

    # Save final checkpoint.
    final_path = os.path.join(checkpoint_dir, f"{task_name}_epoch{history['epoch'][-1]}_final.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": history["epoch"][-1],
            "train_history": history,
        },
        final_path,
    )
    print(f"Final checkpoint saved to {final_path}")

    return history


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from data.modular_addition import get_modular_addition_dataloaders
    from models.transformer import create_standard_transformer

    print("=== Smoke test: modular addition p=7, 200 epochs ===\n")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tiny dataset: p=7 gives 49 examples — fast to train.
    train_loader, test_loader = get_modular_addition_dataloaders(
        p=7, train_frac=0.3, batch_size=16, seed=42
    )

    # Tiny model for speed.
    model = create_standard_transformer(
        task="modular_addition",
        cfg_overrides={"mod_p": 7, "n_layers": 1, "d_model": 32, "d_head": 8, "d_mlp": 64},
        seed=42,
    )

    history = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=200,
        lr=1e-3,
        weight_decay=1.0,
        checkpoint_dir="checkpoints/smoke_test",
        checkpoint_every=100,
        device=device,
        task_name="modular_addition",
    )

    final_train_acc = history["train_acc"][-1]
    print(f"\nFinal train_acc: {final_train_acc:.4f}")
    assert final_train_acc > 0.5, f"Expected train_acc > 0.5 after 200 epochs, got {final_train_acc:.4f}"
    print("Smoke test PASSED.")
