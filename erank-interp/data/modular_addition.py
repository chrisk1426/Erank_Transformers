"""
data/modular_addition.py

Modular addition data pipeline for erank-interp.

Generates all (a, b) pairs where a, b ∈ {0, ..., p-1}, computes (a + b) mod p,
and formats them as token sequences [a, b, EQUALS_TOKEN] with label (a + b) % p.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Module-level constants for default p=113 .
# Functions use the `p` parameter directly to support any modulus.
P: int = 113
EQUALS_TOKEN: int = 113  # one past the last operand token
VOCAB_SIZE: int = 114    # token IDs 0..113


def generate_modular_addition_data(
    p: int = 113,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate all p² (a, b) pairs for modular addition.

    Each example is an input sequence [a, b, p] (where p acts as EQUALS_TOKEN)
    and a label (a + b) % p.

    Args:
        p: The modulus. Operands a, b ∈ {0, ..., p-1}. EQUALS_TOKEN = p.
        seed: Random seed for reproducibility.

    Returns:
        inputs: LongTensor of shape (p*p, 3) containing [a, b, EQUALS_TOKEN] rows.
        labels: LongTensor of shape (p*p,) containing (a + b) % p.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Generate all (a, b) pairs where a, b ∈ {0, ..., p-1}.
    a_vals = torch.arange(p, dtype=torch.long)
    b_vals = torch.arange(p, dtype=torch.long)
    grid_a, grid_b = torch.meshgrid(a_vals, b_vals, indexing="ij")

    # Flatten the grid into a single dimension.
    a_flat = grid_a.reshape(-1)   # shape (p*p,)
    b_flat = grid_b.reshape(-1)   # shape (p*p,)

    # Create a column of EQUALS_TOKEN tokens for each example.
    eq_col = torch.full((p * p,), p, dtype=torch.long)  # EQUALS_TOKEN = p

    # Stack the operands and EQUALS_TOKEN tokens into a single tensor.
    inputs = torch.stack([a_flat, b_flat, eq_col], dim=1)  # shape (p*p, 3)

    # Compute the labels as (a + b) % p for each example.
    labels = (a_flat + b_flat) % p                          # shape (p*p,)

    return inputs, labels


def get_modular_addition_dataloaders(
    p: int = 113,
    train_frac: float = 0.3,
    batch_size: int = 256,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders for the modular addition task.

    Uses a 30/70 train/test split by default, matching Nanda's grokking setup.
    The split is over randomly shuffled (a, b) pairs with a fixed seed.

    Args:
        p: The modulus.
        train_frac: Fraction of examples used for training.
        batch_size: Batch size for both loaders.
        seed: Random seed for the train/test split shuffle.

    Returns:
        (train_loader, test_loader)
    """
    inputs, labels = generate_modular_addition_data(p=p, seed=seed)

    # Calculate the total number of examples.
    n_total = p * p

    # Use a local Generator for the shuffle so we don't pollute global state.
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(n_total, generator=rng)

    # Calculate the number of training examples.
    n_train = int(n_total * train_frac)  # truncation, not rounding

    # Split the examples into training and test sets.
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    # Create PyTorch Datasets for the training and test sets.
    train_dataset = TensorDataset(inputs[train_idx], labels[train_idx])
    test_dataset = TensorDataset(inputs[test_idx], labels[test_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    print("=== Modular Addition Data Verification ===\n")

    # 1. Generate raw data and print shapes.
    inputs, labels = generate_modular_addition_data()
    print(f"inputs shape:  {inputs.shape}  (expected: {113**2}, 3)")
    print(f"labels shape:  {labels.shape}  (expected: {113**2})")
    print(f"inputs dtype:  {inputs.dtype}")
    print(f"labels dtype:  {labels.dtype}\n")

    # 2. Assert label correctness.
    # Extract the operands from the inputs.
    a = inputs[:, 0]
    b = inputs[:, 1]
    assert torch.all(labels == (a + b) % 113), "Label mismatch: labels do not equal (a+b) % p"
    print("Label correctness: PASSED")

    # 3. Assert total count.
    # Assert that the total number of examples is 113^2.
    assert len(inputs) == 113 ** 2, f"Expected {113**2} examples, got {len(inputs)}"
    print(f"Total count ({113**2}): PASSED")

    # 4. Build DataLoaders and check split sizes.
    train_loader, test_loader = get_modular_addition_dataloaders()
    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"\nTrain size: {n_train}  |  Test size: {n_test}  |  Total: {n_train + n_test}")
    assert n_train + n_test == 113 ** 2, "Split sizes do not sum to total"
    print("Split size sum: PASSED")

    # 5. Assert train/test disjointness.
    train_inputs_set = set(map(tuple, train_loader.dataset.tensors[0].tolist()))
    test_inputs_set = set(map(tuple, test_loader.dataset.tensors[0].tolist()))
    assert train_inputs_set.isdisjoint(test_inputs_set), "Train and test sets overlap!"
    print("Train/test disjointness: PASSED")

    print("\nAll assertions passed.")
