"""
analysis/extract_activations.py

Activation extraction pipeline for erank-interp.

Loads trained HookedTransformer checkpoints and extracts residual stream
activations at resid_pre, resid_mid, and resid_post for every layer using
TransformerLens's run_with_cache interface.
"""

from __future__ import annotations

import os
import sys

import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from data.key_value import get_key_value_dataloaders
from data.modular_addition import get_modular_addition_dataloaders
from models.transformer import create_attention_only_transformer, create_standard_transformer


# ---------------------------------------------------------------------------
# Hook name utilities
# ---------------------------------------------------------------------------

def get_hook_names(n_layers: int) -> list[str]:
    """
    Return all hook names for resid_pre, resid_mid, resid_post across all layers.

    Args:
        n_layers: Number of transformer layers.

    Returns:
        List of hook name strings in layer order, e.g.:
        ['blocks.0.hook_resid_pre', 'blocks.0.hook_resid_mid',
         'blocks.0.hook_resid_post', 'blocks.1.hook_resid_pre', ...]
    """
    names: list[str] = []
    for layer in range(n_layers):
        names.append(f"blocks.{layer}.hook_resid_pre")
        names.append(f"blocks.{layer}.hook_resid_mid")
        names.append(f"blocks.{layer}.hook_resid_post")
    return names


def _resid_filter(name: str) -> bool:
    """Names filter for run_with_cache — keeps only residual stream hooks."""
    return (
        "hook_resid_pre" in name
        or "hook_resid_mid" in name
        or "hook_resid_post" in name
    )


def _is_attn_only(model: HookedTransformer) -> bool:
    """Return True if the model has no MLP sublayers (attn_only=True)."""
    return getattr(model.cfg, "attn_only", False)


def _pred_position(task_name: str, n_ctx: int) -> int:
    """Return the sequence position index to extract activations from."""
    if task_name == "modular_addition":
        return 2
    elif task_name == "key_value":
        return n_ctx - 1  # last token
    else:
        raise ValueError(f"Unknown task_name: {task_name!r}")


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_activations(
    model: HookedTransformer,
    dataloader: DataLoader,
    task_name: str,
    n_examples: int = 500,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """
    Run test examples through the model and extract residual stream activations.

    Args:
        model: Trained HookedTransformer (already on device).
        dataloader: DataLoader yielding (inputs, labels) batches.
        task_name: "modular_addition" or "key_value".
        n_examples: Number of test examples to use (stops early if fewer available).
        device: Device to run on.

    Returns:
        Dictionary mapping hook names to activation matrices of shape
        (n_examples, d_model).  Tensors are on CPU.
    """
    torch.manual_seed(42)
    _device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(_device)
    model.eval()

    n_layers: int = model.cfg.n_layers
    d_model: int = model.cfg.d_model
    n_ctx: int = model.cfg.n_ctx
    pos: int = _pred_position(task_name, n_ctx)
    hook_names: list[str] = get_hook_names(n_layers)

    collected = 0

    # For attention-only models hook_resid_mid is absent — we collect pre/post only.
    attn_only: bool = _is_attn_only(model)
    # Names that actually exist in the cache for this model.
    cache_names: list[str] = []
    for layer in range(n_layers):
        cache_names.append(f"blocks.{layer}.hook_resid_pre")
        if not attn_only:
            cache_names.append(f"blocks.{layer}.hook_resid_mid")
        cache_names.append(f"blocks.{layer}.hook_resid_post")

    accum_cache: dict[str, list[torch.Tensor]] = {name: [] for name in cache_names}

    with torch.no_grad():
        for inputs, _labels in dataloader:
            if collected >= n_examples:
                break

            inputs = inputs.to(_device)
            batch_size = inputs.size(0)
            remaining = n_examples - collected
            if batch_size > remaining:
                inputs = inputs[:remaining]
                batch_size = remaining

            _logits, cache = model.run_with_cache(
                inputs,
                names_filter=_resid_filter,
                return_type="logits",
            )

            for name in cache_names:
                # cache[name] shape: (batch, n_ctx, d_model) — take prediction pos
                vec = cache[name][:, pos, :].detach().cpu()  # (batch, d_model)
                accum_cache[name].append(vec)

            collected += batch_size

    # Stack into (n_collected, d_model) matrices.
    n_collected = min(collected, n_examples)
    raw: dict[str, torch.Tensor] = {}
    for name in cache_names:
        raw[name] = torch.cat(accum_cache[name], dim=0)[:n_collected]

    # For attention-only models, synthesize hook_resid_mid = hook_resid_post
    # so that downstream code always sees the full (pre, mid, post) triple.
    # This makes delta_erank_mlp = erank_post - erank_mid = 0 naturally.
    result: dict[str, torch.Tensor] = {}
    for name in hook_names:
        if name in raw:
            result[name] = raw[name]
        else:
            # Must be hook_resid_mid on an attn_only model — use resid_post.
            layer = int(name.split(".")[1])
            result[name] = raw[f"blocks.{layer}.hook_resid_post"].clone()

    # -----------------------------------------------------------------
    # Verification
    # -----------------------------------------------------------------
    for name in hook_names:
        assert name in result, f"Missing hook: {name}"
        shape = result[name].shape
        assert shape == (n_collected, d_model), (
            f"{name}: expected shape ({n_collected}, {d_model}), got {shape}"
        )
        assert not torch.isnan(result[name]).any(), f"NaN in activations for {name}"
        assert not torch.isinf(result[name]).any(), f"Inf in activations for {name}"

    # Control check for attention-only models: resid_mid == resid_post.
    if attn_only:
        for layer in range(n_layers):
            mid = result[f"blocks.{layer}.hook_resid_mid"]
            post = result[f"blocks.{layer}.hook_resid_post"]
            max_diff = (mid - post).abs().max().item()
            print(
                f"  [attn-only control] layer {layer}: "
                f"max |resid_mid - resid_post| = {max_diff:.2e}"
            )

    return result


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def _build_dataloaders_from_cfg(task_name: str, cfg: dict):
    """Reconstruct (train_loader, test_loader) from a saved config dict."""
    if task_name == "modular_addition":
        return get_modular_addition_dataloaders(
            p=cfg["mod_p"],
            train_frac=cfg["mod_train_frac"],
            batch_size=cfg["batch_size"],
            seed=cfg["seed"],
        )
    elif task_name == "key_value":
        return get_key_value_dataloaders(
            num_keys=cfg["kv_num_keys"],
            vocab_size=cfg["kv_vocab_size"],
            train_frac=cfg["kv_train_frac"],
            batch_size=cfg["batch_size"],
            seed=cfg["seed"],
        )
    else:
        raise ValueError(f"Unknown task_name: {task_name!r}")


def load_model_and_extract(
    checkpoint_path: str,
    task_name: str,
    model_type: str,
    n_examples: int = 500,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """
    Convenience function: load a checkpoint, rebuild the model,
    load weights, and extract activations.

    Args:
        checkpoint_path: Path to .pt checkpoint file saved by train_all.py.
        task_name: "modular_addition" or "key_value".
        model_type: "standard" or "attention_only".
        n_examples: Number of examples for extraction.
        device: "cuda" or "cpu".

    Returns:
        Dict mapping hook names to activation matrices (n_examples, d_model) on CPU.
    """
    _device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=_device)
    cfg: dict = ckpt["cfg"]

    # Build cfg_overrides for model factory.
    if task_name == "modular_addition":
        cfg_overrides = {"mod_p": cfg["mod_p"]}
    else:
        cfg_overrides = {
            "kv_num_keys": cfg["kv_num_keys"],
            "kv_vocab_size": cfg["kv_vocab_size"],
        }

    if model_type == "standard":
        model = create_standard_transformer(task_name, cfg_overrides=cfg_overrides, seed=cfg["seed"])
    elif model_type == "attention_only":
        model = create_attention_only_transformer(task_name, cfg_overrides=cfg_overrides, seed=cfg["seed"])
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(_device)
    model.eval()

    _train_loader, test_loader = _build_dataloaders_from_cfg(task_name, cfg)

    print(f"  Loaded {model_type} / {task_name} from {os.path.basename(checkpoint_path)}")
    return extract_activations(model, test_loader, task_name, n_examples=n_examples, device=device)


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")

    for model_type in ("standard", "attention_only"):
        for task_name in ("modular_addition", "key_value"):
            run_name = f"{model_type}_{task_name}"
            ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pt")
            print(f"\n=== {run_name} ===")
            acts = load_model_and_extract(
                ckpt_path, task_name, model_type, n_examples=50, device=device
            )
            for name, mat in acts.items():
                print(f"  {name}: {tuple(mat.shape)}")
    print("\nSmoke test passed.")
