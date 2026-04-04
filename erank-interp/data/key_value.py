"""
data/key_value.py

Key-value retrieval data pipeline for erank-interp.

The model sees a sequence of key-value pairs followed by a query key, and must
output the value associated with that key.

Token encoding:
  - Value tokens: 0 to vocab_size - 1
  - Key tokens:   vocab_size to vocab_size + num_keys - 1
  - Total vocab:  vocab_size + num_keys

Sequence format (length = 2*num_keys + 1):
  [k0, v0, k1, v1, ..., k_{N-1}, v_{N-1}, QUERY_KEY]

The key identity is structural (key k always has token ID vocab_size + k).
Only the key-value bindings are random per example.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Module-level constants for default configuration (documentation anchors only).
# Functions use their own parameters directly.
NUM_KEYS: int = 5
VOCAB_SIZE: int = 30
SEQ_LEN: int = 2 * NUM_KEYS + 1  # = 11


def generate_key_value_data(
    num_keys: int = 5,
    vocab_size: int = 30,
    num_examples: int = 10000,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random key-value retrieval examples.

    For each example, each key is assigned a random value. One key is chosen
    as the query; the label is its associated value.

    Args:
        num_keys: Number of distinct keys (and key-value pairs per sequence).
        vocab_size: Number of value token types (value IDs are 0..vocab_size-1).
        num_examples: Total number of examples to generate.
        seed: Random seed for reproducibility.

    Returns:
        inputs: LongTensor of shape (num_examples, 2*num_keys + 1).
        labels: LongTensor of shape (num_examples,) — value token for the queried key.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Calculate the sequence length.
    seq_len = 2 * num_keys + 1
    key_offset = vocab_size  # key k has token ID vocab_size + k
    
    # Generate random value assignments for each key in each example.

    # Random value assignment: values[i, k] = value token for key k in example i.
    values = torch.randint(0, vocab_size, (num_examples, num_keys), dtype=torch.long)

    # Random query key index for each example.
    query_key_idx = torch.randint(0, num_keys, (num_examples,), dtype=torch.long)

    # Build input sequences.
    inputs = torch.zeros((num_examples, seq_len), dtype=torch.long)
    for k in range(num_keys):
        inputs[:, 2 * k] = key_offset + k   # key token (fixed per slot)
        inputs[:, 2 * k + 1] = values[:, k] # value token (random per example)
    inputs[:, 2 * num_keys] = key_offset + query_key_idx  # query key (last position)

    # Label: the value associated with the queried key.
    labels = values[torch.arange(num_examples), query_key_idx]  # shape (num_examples,)

    return inputs, labels


def get_key_value_dataloaders(
    num_keys: int = 5,
    vocab_size: int = 30,
    num_examples: int = 10000,
    train_frac: float = 0.7,
    batch_size: int = 256,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders for the key-value retrieval task.

    Uses a 70/30 train/test split by default. The split is a random shuffle
    with a fixed seed using a local Generator (does not affect global RNG state).

    Args:
        num_keys: Number of distinct keys per sequence.
        vocab_size: Number of value token types.
        num_examples: Total number of examples to generate.
        train_frac: Fraction of examples used for training.
        batch_size: Batch size for both loaders.
        seed: Random seed for data generation and split shuffle.

    Returns:
        (train_loader, test_loader)
    """
    inputs, labels = generate_key_value_data(
        num_keys=num_keys,
        vocab_size=vocab_size,
        num_examples=num_examples,
        seed=seed,
    )

    n_train = int(num_examples * train_frac)  # truncation, not rounding

    # Local Generator for the shuffle — does not pollute global RNG state.
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(num_examples, generator=rng)


    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    # Create PyTorch Datasets for the training and test sets.
    train_dataset = TensorDataset(inputs[train_idx], labels[train_idx])
    test_dataset = TensorDataset(inputs[test_idx], labels[test_idx])

    # Create DataLoaders for the training and test sets.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    print("=== Key-Value Retrieval Data Verification ===\n")

    num_keys = 5
    vocab_size = 30
    num_examples = 10000
    seq_len = 2 * num_keys + 1  # = 11

    # 1. Generate raw data and print shapes.
    inputs, labels = generate_key_value_data(
        num_keys=num_keys, vocab_size=vocab_size, num_examples=num_examples
    )
    print(f"inputs shape:  {inputs.shape}  (expected: {num_examples}, {seq_len})")
    print(f"labels shape:  {labels.shape}  (expected: {num_examples})")
    print(f"inputs dtype:  {inputs.dtype}")
    print(f"labels dtype:  {labels.dtype}\n")

    # 2. Assert label correctness: label must equal the value at the query key's position.
    query_key_tokens = inputs[:, -1]                          # last column = query key token
    query_key_idx = query_key_tokens - vocab_size             # convert to key index 0..num_keys-1
    value_col_idx = 2 * query_key_idx + 1                    # column in inputs where value lives
    retrieved = inputs[torch.arange(num_examples), value_col_idx]
    assert torch.all(labels == retrieved), "Label mismatch: labels do not match retrieved values"
    print("Label correctness: PASSED")

    # 3. Assert sequence lengths.
    assert inputs.shape[1] == seq_len, f"Expected seq_len={seq_len}, got {inputs.shape[1]}"
    print(f"Sequence length ({seq_len}): PASSED")

    # 4. Assert token bounds.
    total_vocab = vocab_size + num_keys
    assert int(inputs.min()) >= 0 and int(inputs.max()) < total_vocab, (
        f"Input tokens out of bounds [0, {total_vocab})"
    )
    assert int(labels.min()) >= 0 and int(labels.max()) < vocab_size, (
        f"Labels out of bounds [0, {vocab_size})"
    )
    print(f"Token bounds [0, {total_vocab}) and labels [0, {vocab_size}): PASSED")

    # 5. Build DataLoaders and check split sizes.
    train_loader, test_loader = get_key_value_dataloaders(
        num_keys=num_keys, vocab_size=vocab_size, num_examples=num_examples
    )
    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"\nTrain size: {n_train}  |  Test size: {n_test}  |  Total: {n_train + n_test}")
    assert n_train + n_test == num_examples, "Split sizes do not sum to total"
    assert n_train == int(num_examples * 0.7), f"Expected n_train={int(num_examples * 0.7)}, got {n_train}"
    print("Split sizes: PASSED")

    print("\nAll assertions passed.")
