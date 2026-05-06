"""
scripts/run_hybrid_m3_feasibility.py

Feasibility run: hybrid retrieve-then-add with m=3 key-value pairs.

Task: y = (v(q1) + v(q2)) mod 113
  p=113, num_kv_pairs=3, standard architecture, seed=0

With m=3:
  n_ctx = 2*3 + 4 = 10
  d_vocab = 113 + 3 + 2 = 118
  d_vocab_out = 113

Sequence: [K0,V0, K1,V1, K2,V2, QUERY, q1, q2, EQ]

Outputs:
  outputs/hybrid_simplified_m3_standard_seed0/training_log.csv
  outputs/hybrid_simplified_m3_standard_seed0/final_or_best_checkpoint.pt
  outputs/hybrid_simplified_m3_standard_seed0/endpoint_erank.json
  outputs/hybrid_simplified_m3_standard_seed0/hybrid_m3_feasibility_report.md
  outputs/hybrid_simplified_m3_standard_seed0/training.png
"""

from __future__ import annotations

import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from data.splits import get_hybrid_dataloaders_3way
from models.transformer import create_standard_transformer
from training.train_v2 import train_v2
from analysis.extract_activations import extract_activations
from analysis.compute_erank_profiles import compute_erank_profile

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED         = 0
P            = 113
NUM_KV_PAIRS = 3          # simplified: 3 instead of 4
MAX_EPOCHS   = 50000
OUT_DIR      = os.path.join(PROJECT_ROOT, "outputs", "hybrid_simplified_m3_standard_seed0")
FEASIBILITY_THRESHOLD = 0.05   # val_acc > 5% = promising


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load base config for shared hyperparameters
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Confirm expected dims
    n_ctx     = 2 * NUM_KV_PAIRS + 4   # = 10
    d_vocab   = P + NUM_KV_PAIRS + 2   # = 118
    d_vocab_out = P                    # = 113
    print(f"Hybrid m={NUM_KV_PAIRS}: n_ctx={n_ctx} d_vocab={d_vocab} d_vocab_out={d_vocab_out}")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print("Building dataloaders...")
    train_loader, val_loader, test_loader, split_meta = get_hybrid_dataloaders_3way(
        p=P,
        num_kv_pairs=NUM_KV_PAIRS,
        num_examples=cfg.get("hybrid_num_examples", 20000),
        train_frac=cfg.get("hybrid_train_frac", 0.7),
        batch_size=cfg["batch_size"],
        seed=SEED,
    )
    print(f"  train={split_meta['n_train']} val={split_meta['n_val']} test={split_meta['n_test']}")

    split_path = os.path.join(OUT_DIR, "split_metadata.json")
    with open(split_path, "w") as f:
        json.dump(split_meta, f, indent=2)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    torch.manual_seed(SEED)
    model = create_standard_transformer(
        "hybrid_retrieve_add",
        cfg_overrides={"hybrid_p": P, "hybrid_num_kv_pairs": NUM_KV_PAIRS},
        seed=SEED,
    )
    assert model.cfg.n_ctx == n_ctx
    assert model.cfg.d_vocab == d_vocab
    assert model.cfg.d_vocab_out == d_vocab_out
    print(f"Model: n_ctx={model.cfg.n_ctx} d_vocab={model.cfg.d_vocab} "
          f"d_vocab_out={model.cfg.d_vocab_out}")

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    ckpt_dir = os.path.join(OUT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Training for up to {MAX_EPOCHS} epochs...")
    history = train_v2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=MAX_EPOCHS,
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        checkpoint_dir=ckpt_dir,
        checkpoint_name="model",
        device=device,
        task_name="hybrid_retrieve_add",
    )

    best_val_epoch = history["best_val_epoch"]
    best_val_acc   = history["best_val_acc"]
    # Test acc at best val epoch
    if best_val_epoch and best_val_epoch in history["epoch"]:
        idx = history["epoch"].index(best_val_epoch)
        test_acc_at_bv = history["test_acc"][idx]
    else:
        test_acc_at_bv = history["test_acc"][-1] if history["test_acc"] else None

    final_train_acc = history["train_acc"][-1] if history["train_acc"] else None
    stopping_epoch  = history["epoch"][-1] if history["epoch"] else MAX_EPOCHS

    print(f"\nTraining complete.")
    print(f"  best_val_epoch={best_val_epoch}  best_val_acc={best_val_acc:.4f}")
    print(f"  test_acc@best_val={test_acc_at_bv}")
    print(f"  final train_acc={final_train_acc:.4f}  stopping_epoch={stopping_epoch}")

    # -----------------------------------------------------------------------
    # Save training log CSV
    # -----------------------------------------------------------------------
    log_csv = os.path.join(OUT_DIR, "training_log.csv")
    with open(log_csv, "w", newline="") as f:
        fieldnames = ["epoch", "train_loss", "val_loss", "test_loss",
                      "train_acc", "val_acc", "test_acc"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, ep in enumerate(history["epoch"]):
            writer.writerow({
                "epoch":      ep,
                "train_loss": history["train_loss"][i] if i < len(history["train_loss"]) else "",
                "val_loss":   history["val_loss"][i]   if i < len(history["val_loss"])   else "",
                "test_loss":  history["test_loss"][i]  if i < len(history["test_loss"])  else "",
                "train_acc":  history["train_acc"][i]  if i < len(history["train_acc"])  else "",
                "val_acc":    history["val_acc"][i]    if i < len(history["val_acc"])    else "",
                "test_acc":   history["test_acc"][i]   if i < len(history["test_acc"])   else "",
            })
    print(f"Saved: {log_csv}")

    # -----------------------------------------------------------------------
    # Copy best-val checkpoint as final_or_best_checkpoint.pt
    # -----------------------------------------------------------------------
    best_ckpt = os.path.join(ckpt_dir, "model_best_val.pt")
    final_ckpt = os.path.join(OUT_DIR, "final_or_best_checkpoint.pt")
    if os.path.exists(best_ckpt):
        import shutil
        shutil.copy2(best_ckpt, final_ckpt)
        print(f"Saved: {final_ckpt}")

    # -----------------------------------------------------------------------
    # Endpoint eRank (on best-val checkpoint)
    # -----------------------------------------------------------------------
    erank_result = None
    if os.path.exists(best_ckpt):
        try:
            print("Computing endpoint eRank...")
            ckpt_data = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            model_for_erank = create_standard_transformer(
                "hybrid_retrieve_add",
                cfg_overrides={"hybrid_p": P, "hybrid_num_kv_pairs": NUM_KV_PAIRS},
                seed=SEED,
            )
            model_for_erank.load_state_dict(ckpt_data["model_state_dict"])
            acts = extract_activations(
                model_for_erank, test_loader, "hybrid_retrieve_add",
                n_examples=500, device=device
            )
            erank_result = compute_erank_profile(acts, n_layers=model_for_erank.cfg.n_layers)

            # Compute sums and layer deltas
            da = erank_result.get("delta_erank_attn", {})
            dm = erank_result.get("delta_erank_mlp", {})
            sum_da = sum(da.values()) if da else 0.0
            sum_dm = sum(dm.values()) if dm else 0.0
            erank_result["sum_delta_attn"] = sum_da
            erank_result["sum_delta_mlp"]  = sum_dm

            erank_path = os.path.join(OUT_DIR, "endpoint_erank.json")
            with open(erank_path, "w") as f:
                json.dump(erank_result, f, indent=2)
            print(f"  sum_delta_attn={sum_da:.4f}  sum_delta_mlp={sum_dm:.4f}")
            print(f"Saved: {erank_path}")
        except Exception as e:
            print(f"  eRank computation failed: {e}")

    # -----------------------------------------------------------------------
    # Training curve plot
    # -----------------------------------------------------------------------
    _plot_training(history, os.path.join(OUT_DIR, "training.png"))

    # -----------------------------------------------------------------------
    # Feasibility report
    # -----------------------------------------------------------------------
    _write_report(
        out_dir=OUT_DIR,
        best_val_epoch=best_val_epoch,
        best_val_acc=best_val_acc,
        test_acc_at_bv=test_acc_at_bv,
        final_train_acc=final_train_acc,
        stopping_epoch=stopping_epoch,
        erank_result=erank_result,
        split_meta=split_meta,
        feasibility_threshold=FEASIBILITY_THRESHOLD,
        p=P, m=NUM_KV_PAIRS, max_epochs=MAX_EPOCHS,
    )

    print(f"\nAll outputs saved to: {OUT_DIR}")


def _plot_training(history: dict, save_path: str):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = history["epoch"]

    if history.get("train_loss"):
        ax_loss.plot(epochs, history["train_loss"], label="train")
    if history.get("val_loss"):
        ax_loss.plot(epochs, history["val_loss"],   label="val")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Hybrid m=3 — Loss"); ax_loss.legend()

    if history.get("train_acc"):
        ax_acc.plot(epochs, history["train_acc"],  label="train")
    if history.get("val_acc"):
        ax_acc.plot(epochs, history["val_acc"],    label="val")
    if history.get("test_acc"):
        ax_acc.plot(epochs, history["test_acc"],   label="test", linestyle="--", alpha=0.6)
    ax_acc.axhline(1/113, color="red", linestyle=":", linewidth=1, label="chance (1/113)")
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_title("Hybrid m=3 — Accuracy"); ax_acc.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def _write_report(out_dir, best_val_epoch, best_val_acc, test_acc_at_bv,
                  final_train_acc, stopping_epoch, erank_result,
                  split_meta, feasibility_threshold, p, m, max_epochs):
    lines = []
    a = lines.append
    chance = 1.0 / p

    a("# Hybrid m=3 Feasibility Report")
    a("")
    a(f"**Task:** hybrid_retrieve_add, p={p}, num_kv_pairs={m}")
    a(f"**Architecture:** standard  |  **Seed:** 0")
    a(f"**Max epochs:** {max_epochs}")
    a("")
    a("## Task description")
    a(f"- Sequence length: {2*m+4}, d_vocab: {p+m+2}, d_vocab_out: {p}")
    a(f"- {split_meta.get('n_train','?')} train / {split_meta.get('n_val','?')} val / {split_meta.get('n_test','?')} test examples")
    a(f"- Chance accuracy: 1/{p} ≈ {chance:.4f}")
    a("")
    a("## Results")
    a("")
    a(f"| Metric | Value |")
    a(f"|---|---|")
    a(f"| Best val epoch | {best_val_epoch} |")
    a(f"| Best val acc | {best_val_acc:.4f} |")
    a(f"| Test acc @ best val | {test_acc_at_bv:.4f} if test_acc_at_bv is not None else 'N/A' |")
    a(f"| Final train acc | {final_train_acc:.4f} if final_train_acc is not None else 'N/A' |")
    a(f"| Stopping epoch | {stopping_epoch} |")
    a(f"| Chance accuracy | {chance:.4f} |")
    a(f"| Feasibility threshold | {feasibility_threshold:.4f} |")
    a("")

    if best_val_acc is not None and best_val_acc >= feasibility_threshold:
        a(f"## Verdict: PROMISING")
        a(f"Best val_acc ({best_val_acc:.4f}) exceeds feasibility threshold ({feasibility_threshold:.4f}).")
        a("The model is learning the task. Consider running all 3 seeds and attention-only.")
    elif best_val_acc is not None and best_val_acc > chance * 3:
        a(f"## Verdict: PARTIAL LEARNING")
        a(f"Best val_acc ({best_val_acc:.4f}) is above chance ({chance:.4f}) but below threshold ({feasibility_threshold:.4f}).")
        a("Weak generalisation signal. May need more epochs or different hyperparameters.")
    else:
        a(f"## Verdict: FAILED — no generalisation")
        a(f"Best val_acc ({best_val_acc:.4f}) is near chance ({chance:.4f}).")
        a("The m=3 simplification did not help. Consider p=7, m=2 or architectural changes.")

    if erank_result:
        da = erank_result.get("delta_erank_attn", {})
        dm = erank_result.get("delta_erank_mlp", {})
        sum_da = erank_result.get("sum_delta_attn", sum(da.values()) if da else 0)
        sum_dm = erank_result.get("sum_delta_mlp",  sum(dm.values()) if dm else 0)
        a("")
        a("## Endpoint eRank (best-val checkpoint)")
        a("")
        a("| Component | Layer 0 | Layer 1 | Sum |")
        a("|---|---:|---:|---:|")
        a(f"| Δ_attn | {da.get('layer_0', 0):.3f} | {da.get('layer_1', 0):.3f} | {sum_da:.3f} |")
        a(f"| Δ_mlp  | {dm.get('layer_0', 0):.3f} | {dm.get('layer_1', 0):.3f} | {sum_dm:.3f} |")

    report_path = os.path.join(out_dir, "hybrid_m3_feasibility_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
