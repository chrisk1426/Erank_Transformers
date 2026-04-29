"""
models/transformer.py

Model configuration and factory functions for erank-interp.

Creates tiny HookedTransformers via TransformerLens for both tasks:
  - modular_addition: context [a, b, =], vocab 114, output 113
  - key_value:        context [k0,v0,...,kN,vN,QUERY], vocab vocab_size+num_keys
"""

from __future__ import annotations

import random

import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig


def _get_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_cfg_dict(task: str, seed: int, cfg_overrides: dict) -> dict:
    """
    Build a HookedTransformerConfig-compatible dict for the given task.

    Task-specific keys (mod_p, kv_num_keys, kv_vocab_size) are popped from
    cfg_overrides before it is merged, so unknown keys never reach
    HookedTransformerConfig.

    Args:
        task: "modular_addition" or "key_value".
        seed: Random seed passed into the config.
        cfg_overrides: Caller-supplied overrides (mutated in-place via pop —
            callers must pass a copy).

    Returns:
        A plain dict ready for HookedTransformerConfig(**cfg_dict).
    """
    device = _get_device()

    # modular addition
    # n_ctx = 3
    # d_vocab = p + 1
    # d_vocab_out = p

    # key-value retrieval
    # n_ctx = 2 * num_keys + 1
    # d_vocab = vocab_size + num_keys
    # d_vocab_out = vocab_size
    if task == "modular_addition":
        p = cfg_overrides.pop("mod_p", 113)
        task_fields = dict(
            n_ctx=3,
            d_vocab=p + 1,       # tokens 0..p-1 (operands) + p (EQUALS_TOKEN)
            d_vocab_out=p,       # output logits over 0..p-1
        )
    elif task == "key_value":
        num_keys = cfg_overrides.pop("kv_num_keys", 5)
        vocab_size = cfg_overrides.pop("kv_vocab_size", 30)
        task_fields = dict(
            n_ctx=2 * num_keys + 1,
            d_vocab=vocab_size + num_keys,
            d_vocab_out=vocab_size,
        )
    else:
        raise ValueError(f"Unknown task: {task!r}. Expected 'modular_addition' or 'key_value'.")

    # default config
    # n_layers = 2
    # d_model = 128
    # n_heads = 4
    # d_head = 32
    # d_mlp = 512
    # act_fn = "gelu"
    # normalization_type = None
    # seed = seed
    # device = device
    # **task_fields
    cfg_dict = dict(
        n_layers=2,
        d_model=128,
        n_heads=4,
        d_head=32,
        d_mlp=512,
        act_fn="gelu",
        normalization_type=None,
        seed=seed,
        device=device,
        **task_fields,
    )

    # Apply caller overrides last so they can change any field.
    cfg_dict.update(cfg_overrides)
    return cfg_dict


def create_standard_transformer(
    task: str,
    cfg_overrides: dict | None = None,
    seed: int = 42,
) -> HookedTransformer:
    """
    Create a standard (full) HookedTransformer for the given task.

    Args:
        task: "modular_addition" or "key_value".
        cfg_overrides: Optional dict of config field overrides. Task-specific
            keys (mod_p, kv_num_keys, kv_vocab_size) are handled internally
            and must not be passed as HookedTransformerConfig fields.
        seed: Random seed for weight initialisation and RNG state.

    Returns:
        Untrained HookedTransformer configured for the task.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    overrides = dict(cfg_overrides or {})
    cfg_dict = _make_cfg_dict(task, seed, overrides)
    cfg = HookedTransformerConfig(**cfg_dict)
    return HookedTransformer(cfg)


def create_attention_only_transformer(
    task: str,
    cfg_overrides: dict | None = None,
    seed: int = 42,
) -> HookedTransformer:
    """
    Create an attention-only HookedTransformer (no MLP blocks) for the given task.

    Sets attn_only=True, which disables the MLP sublayer in every layer.
    d_mlp is removed from the config dict because TransformerLens ignores it
    when attn_only=True, and passing it can cause unexpected behaviour.

    Args:
        task: "modular_addition" or "key_value".
        cfg_overrides: Optional config overrides (same semantics as above).
        seed: Random seed for weight initialisation and RNG state.

    Returns:
        Untrained attention-only HookedTransformer configured for the task.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    overrides = dict(cfg_overrides or {})
    overrides["attn_only"] = True
    cfg_dict = _make_cfg_dict(task, seed, overrides)
    cfg_dict.pop("d_mlp", None)   # irrelevant and potentially confusing with attn_only=True
    cfg = HookedTransformerConfig(**cfg_dict)
    return HookedTransformer(cfg)


if __name__ == "__main__":
    TASKS = ["modular_addition", "key_value"]

    # Expected config values per task for verification.
    EXPECTED = {
        "modular_addition": dict(n_ctx=3, d_vocab=114, d_vocab_out=113),
        "key_value":        dict(n_ctx=11, d_vocab=35,  d_vocab_out=30),
    }

    for task in TASKS:
        print(f"\n=== {task} ===")

        # 1. Instantiation
        std  = create_standard_transformer(task)
        attn = create_attention_only_transformer(task)
        print("  Instantiation: PASSED")

        # 2. Config correctness
        exp = EXPECTED[task]
        for field, val in exp.items():
            assert getattr(std.cfg, field) == val, (
                f"standard {task}: expected {field}={val}, got {getattr(std.cfg, field)}"
            )
            assert getattr(attn.cfg, field) == val, (
                f"attn_only {task}: expected {field}={val}, got {getattr(attn.cfg, field)}"
            )
        print(f"  Config correctness {exp}: PASSED")

        # 3. Parameter counts
        n_std  = sum(p.numel() for p in std.parameters())
        n_attn = sum(p.numel() for p in attn.parameters())
        assert n_std > n_attn, "Standard model should have more params than attention-only"
        print(f"  Params — standard: {n_std:,}  attention-only: {n_attn:,}  PASSED")

        # 4. Forward pass — output shape should be (batch, n_ctx, d_vocab_out)
        batch_size = 4
        dummy = torch.zeros(batch_size, std.cfg.n_ctx, dtype=torch.long)
        with torch.no_grad():
            out_std  = std(dummy)
            out_attn = attn(dummy)
        expected_shape = (batch_size, std.cfg.n_ctx, std.cfg.d_vocab_out)
        assert out_std.shape  == expected_shape, f"standard output shape mismatch: {out_std.shape}"
        assert out_attn.shape == expected_shape, f"attn_only output shape mismatch: {out_attn.shape}"
        print(f"  Forward pass output shape {expected_shape}: PASSED")

    print("\nAll checks passed.")
