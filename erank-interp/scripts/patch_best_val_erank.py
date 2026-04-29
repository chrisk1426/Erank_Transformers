"""
scripts/patch_best_val_erank.py

Targeted patch: compute eRank for the standard_modular_addition best-val
checkpoint (epoch 5886) and insert it into the training-time series, then
regenerate the affected CSV/plots/report via run_step4_training_time.

This script should be run once. It is idempotent: re-running it will
overwrite the inserted entry but will not duplicate it.
"""

from __future__ import annotations

import json
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from analysis.extract_activations import extract_activations
from analysis.compute_erank_profiles import compute_erank_profile
from data.splits import get_modular_addition_dataloaders_3way
from models.transformer import create_standard_transformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_ROOT   = os.path.join(PROJECT_ROOT, "outputs", "control_rerun")
CKPT_PATH     = os.path.join(OUTPUT_ROOT, "checkpoints", "standard_modular_addition_best_val.pt")
CURVES_PATH   = os.path.join(OUTPUT_ROOT, "results", "standard_modular_addition_curves.json")

RUN_NAME      = "standard_modular_addition"
N_EXAMPLES    = 500


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load checkpoint and confirm epoch
    # ------------------------------------------------------------------
    print(f"\nLoading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    epoch = ckpt["epoch"]
    cfg   = ckpt["cfg"]
    print(f"  Confirmed epoch: {epoch}")

    # Get acc values from the stored training history
    history = ckpt.get("train_history", {})
    epochs_list  = history.get("epoch", [])
    val_accs     = history.get("val_acc", [])
    train_accs   = history.get("train_acc", [])
    test_accs    = history.get("test_acc", [])

    if epoch in epochs_list:
        idx = epochs_list.index(epoch)
        train_acc = train_accs[idx]
        val_acc   = val_accs[idx]
        test_acc  = test_accs[idx]
    else:
        # Fallback: use the last recorded values
        train_acc = train_accs[-1] if train_accs else float("nan")
        val_acc   = val_accs[-1]   if val_accs   else float("nan")
        test_acc  = test_accs[-1]  if test_accs  else float("nan")

    print(f"  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

    # ------------------------------------------------------------------
    # 2. Rebuild model and load weights
    # ------------------------------------------------------------------
    model = create_standard_transformer(
        "modular_addition",
        cfg_overrides={"mod_p": cfg["mod_p"]},
        seed=cfg["seed"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Model rebuilt: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

    # ------------------------------------------------------------------
    # 3. Reconstruct val_loader using the same settings as training
    # ------------------------------------------------------------------
    _, val_loader, _, _ = get_modular_addition_dataloaders_3way(
        p=cfg["mod_p"],
        train_frac=cfg["mod_train_frac"],
        val_frac_of_train=cfg["val_frac_of_train"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
    )
    print(f"  Val loader: {len(val_loader.dataset)} examples, batch_size={cfg['batch_size']}")

    # ------------------------------------------------------------------
    # 4. Extract activations and compute eRank profile
    # ------------------------------------------------------------------
    print(f"\nExtracting activations from {N_EXAMPLES} val examples...")
    acts = extract_activations(
        model, val_loader, "modular_addition",
        n_examples=N_EXAMPLES, device=str(device),
    )
    profile = compute_erank_profile(acts, n_layers=model.cfg.n_layers)

    n_layers = model.cfg.n_layers
    print("\neRank profile at best-val checkpoint:")
    for l in range(n_layers):
        lk = f"layer_{l}"
        e  = profile["erank"][lk]
        da = profile["delta_erank_attn"][lk]
        dm = profile["delta_erank_mlp"][lk]
        print(
            f"  Layer {l}: pre={e['pre']:.4f}  mid={e['mid']:.4f}  post={e['post']:.4f}"
            f"  Δ_attn={da:+.4f}  Δ_mlp={dm:+.4f}"
        )

    # ------------------------------------------------------------------
    # 5. Build new time-series entry
    # ------------------------------------------------------------------
    new_entry: dict = {
        "epoch":      epoch,
        "train_acc":  train_acc,
        "val_acc":    val_acc,
        "test_acc":   test_acc,
        "checkpoint_type": "best_val",
    }
    for l in range(n_layers):
        lk = f"layer_{l}"
        new_entry[f"layer_{l}_pre"]        = profile["erank"][lk]["pre"]
        new_entry[f"layer_{l}_mid"]        = profile["erank"][lk]["mid"]
        new_entry[f"layer_{l}_post"]       = profile["erank"][lk]["post"]
        new_entry[f"layer_{l}_delta_attn"] = profile["delta_erank_attn"][lk]
        new_entry[f"layer_{l}_delta_mlp"]  = profile["delta_erank_mlp"][lk]

    # ------------------------------------------------------------------
    # 6. Load curves JSON, remove any existing entry at this epoch, insert
    # ------------------------------------------------------------------
    print(f"\nUpdating curves JSON: {CURVES_PATH}")
    with open(CURVES_PATH) as f:
        curves = json.load(f)

    series = curves.get("erank_time_series", [])
    before = len(series)

    # Remove any prior entry at this epoch (idempotent re-runs)
    series = [e for e in series if e["epoch"] != epoch]

    # Insert and sort by epoch
    series.append(new_entry)
    series.sort(key=lambda e: e["epoch"])
    curves["erank_time_series"] = series

    after = len(series)
    print(f"  Time-series entries: {before} → {after}")
    print(f"  Inserted epoch={epoch} (checkpoint_type=best_val)")

    with open(CURVES_PATH, "w") as f:
        json.dump(curves, f, indent=2)
    print("  Saved.")

    # ------------------------------------------------------------------
    # 7. Regenerate CSV / plots / report via run_step4_training_time
    # ------------------------------------------------------------------
    print("\nRegenerating training-time outputs (Step 4)...")
    import importlib.util, types

    step4_path = os.path.join(SCRIPT_DIR, "run_step4_training_time.py")
    spec = importlib.util.spec_from_file_location("run_step4_training_time", step4_path)
    step4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step4)
    step4.main()

    print("\n=== Patch complete ===")
    print(f"  Checkpoint: {CKPT_PATH}")
    print(f"  Epoch inserted: {epoch}")
    print(f"  val_acc at insertion: {val_acc:.4f}")
    print("  Artifacts regenerated by run_step4_training_time.")


if __name__ == "__main__":
    main()
