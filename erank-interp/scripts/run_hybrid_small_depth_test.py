"""
scripts/run_hybrid_small_depth_test.py

Depth feasibility test for hybrid retrieve-then-add with p=23, m=3.
Uses EXHAUSTIVE dataset (23^3 * 6 = 73,002 examples) with 80/10/10 split.

Usage:
    python scripts/run_hybrid_small_depth_test.py --n_layers 3
    python scripts/run_hybrid_small_depth_test.py --n_layers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from data.splits import get_hybrid_dataloaders_3way_exhaustive
from models.transformer import create_standard_transformer
from training.train_v2 import train_v2
from analysis.extract_activations import extract_activations
from analysis.compute_erank_profiles import compute_erank_profile

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED         = 0
P            = 23
NUM_KV_PAIRS = 3
D_MODEL      = 128
N_HEADS      = 4
D_MLP        = 512
LR           = 0.0003
WEIGHT_DECAY = 1.0
BATCH_SIZE   = 256
MAX_EPOCHS   = 8000       # ~73k examples, larger dataset -> fewer epochs needed


def label_sanity_check(train_loader, p, num_kv_pairs):
    """Verify labels for 20 examples. Returns (pass_bool, report_lines)."""
    lines = []
    lines.append("=" * 60)
    lines.append("LABEL SANITY CHECK - 20 examples")
    lines.append("=" * 60)

    batch_x, batch_y = next(iter(train_loader))
    n_check = min(20, batch_x.shape[0])
    all_pass = True

    for i in range(n_check):
        seq = batch_x[i].tolist()
        label = batch_y[i].item()

        kv_pairs = {}
        for j in range(num_kv_pairs):
            k = seq[2 * j]
            v = seq[2 * j + 1]
            kv_pairs[k] = v

        q1_key = seq[2 * num_kv_pairs + 1]
        q2_key = seq[2 * num_kv_pairs + 2]

        v_q1 = kv_pairs.get(q1_key, None)
        v_q2 = kv_pairs.get(q2_key, None)

        if v_q1 is not None and v_q2 is not None:
            expected = (v_q1 + v_q2) % p
        else:
            expected = None

        ok = (expected == label)
        if not ok:
            all_pass = False

        lines.append(f"\nExample {i+1}:")
        lines.append(f"  context: {seq}")
        lines.append(f"  q1_key: {q1_key}  q2_key: {q2_key}")
        lines.append(f"  v(q1): {v_q1}  v(q2): {v_q2}")
        lines.append(f"  expected label: {expected}")
        lines.append(f"  actual label:   {label}")
        lines.append(f"  {'PASS' if ok else 'FAIL'}")

    lines.append("\n" + "=" * 60)
    verdict = "ALL 20 PASSED" if all_pass else "SOME FAILED"
    lines.append(f"VERDICT: {verdict}")
    lines.append("=" * 60)

    return all_pass, lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_layers", type=int, required=True, choices=[3, 4])
    args = parser.parse_args()

    n_layers = args.n_layers

    out_dir = os.path.join(
        PROJECT_ROOT, "outputs",
        f"hybrid_small_p23_m3_standard_{n_layers}layer_seed{SEED}"
    )
    shared_dir = os.path.join(PROJECT_ROOT, "outputs", "hybrid_small_p23_m3_depth_test")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(shared_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Hybrid depth test: p={P} m={NUM_KV_PAIRS} n_layers={n_layers}")

    with open(os.path.join(out_dir, "status.txt"), "w") as f:
        f.write("RUNNING\n")

    n_ctx = 2 * NUM_KV_PAIRS + 4       # = 10
    d_vocab = P + NUM_KV_PAIRS + 2     # = 28
    d_vocab_out = P                     # = 23
    chance = 1.0 / P
    print(f"n_ctx={n_ctx} d_vocab={d_vocab} d_vocab_out={d_vocab_out} n_layers={n_layers}")
    print(f"Chance accuracy: 1/{P} = {chance:.4f}")

    # -----------------------------------------------------------------------
    # Data (exhaustive: 23^3 * 6 = 73,002 examples, 80/10/10 split)
    # -----------------------------------------------------------------------
    print("Building exhaustive dataloaders...")
    train_loader, val_loader, test_loader, split_meta = get_hybrid_dataloaders_3way_exhaustive(
        p=P,
        num_kv_pairs=NUM_KV_PAIRS,
        train_frac=0.8,
        val_frac=0.1,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )
    print(f"  total={split_meta['n_total']} train={split_meta['n_train']} "
          f"val={split_meta['n_val']} test={split_meta['n_test']}")

    split_path = os.path.join(out_dir, "split_metadata.json")
    with open(split_path, "w") as f:
        json.dump(split_meta, f, indent=2)

    # -----------------------------------------------------------------------
    # Label sanity check
    # -----------------------------------------------------------------------
    print("\nRunning label sanity check...")
    sanity_pass, sanity_lines = label_sanity_check(train_loader, P, NUM_KV_PAIRS)
    sanity_text = "\n".join(sanity_lines)
    print(sanity_text)

    sanity_path = os.path.join(shared_dir, "label_sanity_check.txt")
    with open(sanity_path, "w") as f:
        f.write(sanity_text)
    # Also save per-run copy
    with open(os.path.join(out_dir, "label_sanity_check.txt"), "w") as f:
        f.write(sanity_text)

    if not sanity_pass:
        print("ERROR: Label sanity check FAILED. Aborting.")
        with open(os.path.join(out_dir, "status.txt"), "w") as f:
            f.write("FAILED\n")
        sys.exit(1)

    print("Label sanity check PASSED. Proceeding to training.\n")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    torch.manual_seed(SEED)
    model = create_standard_transformer(
        "hybrid_retrieve_add",
        cfg_overrides={
            "hybrid_p": P,
            "hybrid_num_kv_pairs": NUM_KV_PAIRS,
            "n_layers": n_layers,
        },
        seed=SEED,
    )
    assert model.cfg.n_layers == n_layers
    assert model.cfg.n_ctx == n_ctx
    assert model.cfg.d_vocab == d_vocab
    assert model.cfg.d_vocab_out == d_vocab_out
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: n_layers={model.cfg.n_layers} n_ctx={model.cfg.n_ctx} "
          f"d_model={model.cfg.d_model} n_heads={model.cfg.n_heads} "
          f"d_vocab={model.cfg.d_vocab} d_vocab_out={model.cfg.d_vocab_out}")
    print(f"Parameters: {n_params:,}")

    config_used = {
        "task": "hybrid_retrieve_add_small",
        "p": P,
        "num_kv_pairs": NUM_KV_PAIRS,
        "architecture": "standard",
        "n_layers": n_layers,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "d_mlp": D_MLP,
        "n_ctx": n_ctx,
        "d_vocab": d_vocab,
        "d_vocab_out": d_vocab_out,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "dataset": "exhaustive",
        "n_total": split_meta["n_total"],
        "n_train": split_meta["n_train"],
        "n_val": split_meta["n_val"],
        "n_test": split_meta["n_test"],
        "train_frac": 0.8,
        "val_frac": 0.1,
        "seed": SEED,
        "n_params": n_params,
    }
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.dump(config_used, f, default_flow_style=False)

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Training for up to {MAX_EPOCHS} epochs...")
    t_start = time.time()
    history = train_v2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=MAX_EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        checkpoint_dir=ckpt_dir,
        checkpoint_name="model",
        device=device,
        task_name="hybrid_retrieve_add",
    )
    t_elapsed = time.time() - t_start

    best_val_epoch = history["best_val_epoch"]
    best_val_acc = history["best_val_acc"]
    if best_val_epoch and best_val_epoch in history["epoch"]:
        idx = history["epoch"].index(best_val_epoch)
        test_acc_at_bv = history["test_acc"][idx]
    else:
        test_acc_at_bv = history["test_acc"][-1] if history["test_acc"] else None

    final_train_acc = history["train_acc"][-1] if history["train_acc"] else None
    final_val_acc = history["val_acc"][-1] if history["val_acc"] else None
    final_test_acc = history["test_acc"][-1] if history["test_acc"] else None
    stopping_epoch = history["epoch"][-1] if history["epoch"] else MAX_EPOCHS

    print(f"\nTraining complete in {t_elapsed/3600:.1f} hours.")
    print(f"  best_val_epoch={best_val_epoch}  best_val_acc={best_val_acc:.4f}")
    print(f"  test_acc@best_val={test_acc_at_bv}")
    print(f"  final train_acc={final_train_acc:.4f}  stopping_epoch={stopping_epoch}")

    # -----------------------------------------------------------------------
    # Save training log CSV
    # -----------------------------------------------------------------------
    log_csv = os.path.join(out_dir, "training_log.csv")
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
    # Copy best-val checkpoint
    # -----------------------------------------------------------------------
    best_ckpt = os.path.join(ckpt_dir, "model_best_val.pt")
    best_val_ckpt_path = os.path.join(out_dir, "best_val_checkpoint.pt")
    final_ckpt_path = os.path.join(out_dir, "final_checkpoint.pt")

    if os.path.exists(best_ckpt):
        shutil.copy2(best_ckpt, best_val_ckpt_path)
        print(f"Saved: {best_val_ckpt_path}")

    final_ckpt_src = os.path.join(ckpt_dir, "model_final.pt")
    if os.path.exists(final_ckpt_src):
        shutil.copy2(final_ckpt_src, final_ckpt_path)
    else:
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": stopping_epoch,
        }, final_ckpt_path)
    print(f"Saved: {final_ckpt_path}")

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
                cfg_overrides={
                    "hybrid_p": P,
                    "hybrid_num_kv_pairs": NUM_KV_PAIRS,
                    "n_layers": n_layers,
                },
                seed=SEED,
            )
            model_for_erank.load_state_dict(ckpt_data["model_state_dict"])
            acts = extract_activations(
                model_for_erank, test_loader, "hybrid_retrieve_add",
                n_examples=500, device=device
            )
            erank_result = compute_erank_profile(acts, n_layers=n_layers)

            da = erank_result.get("delta_erank_attn", {})
            dm = erank_result.get("delta_erank_mlp", {})
            sum_da = sum(da.values()) if da else 0.0
            sum_dm = sum(dm.values()) if dm else 0.0
            erank_result["sum_delta_attn"] = sum_da
            erank_result["sum_delta_mlp"] = sum_dm

            erank_csv_path = os.path.join(out_dir, "endpoint_erank.csv")
            with open(erank_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["layer", "erank_pre", "erank_mid", "erank_post",
                                 "delta_attn", "delta_mlp"])
                pre = erank_result.get("erank_pre", {})
                mid = erank_result.get("erank_mid", {})
                post = erank_result.get("erank_post", {})
                for layer in range(n_layers):
                    lk = f"layer_{layer}"
                    writer.writerow([
                        layer,
                        pre.get(lk, ""),
                        mid.get(lk, ""),
                        post.get(lk, ""),
                        da.get(lk, ""),
                        dm.get(lk, ""),
                    ])
                writer.writerow([])
                writer.writerow(["sum_delta_attn", sum_da])
                writer.writerow(["sum_delta_mlp", sum_dm])

            erank_json_path = os.path.join(out_dir, "endpoint_erank.json")
            with open(erank_json_path, "w") as f:
                json.dump(erank_result, f, indent=2)

            print(f"  sum_delta_attn={sum_da:.4f}  sum_delta_mlp={sum_dm:.4f}")
            print(f"Saved: {erank_csv_path}")
            print(f"Saved: {erank_json_path}")
        except Exception as e:
            print(f"  eRank computation failed: {e}")
            import traceback
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # Training curve plot
    # -----------------------------------------------------------------------
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = history["epoch"]

    if history.get("train_loss"):
        ax_loss.plot(epochs, history["train_loss"], label="train")
    if history.get("val_loss"):
        ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"Hybrid p={P} m={NUM_KV_PAIRS} {n_layers}L - Loss")
    ax_loss.legend()

    if history.get("train_acc"):
        ax_acc.plot(epochs, history["train_acc"], label="train")
    if history.get("val_acc"):
        ax_acc.plot(epochs, history["val_acc"], label="val")
    if history.get("test_acc"):
        ax_acc.plot(epochs, history["test_acc"], label="test", linestyle="--", alpha=0.6)
    ax_acc.axhline(1/P, color="red", linestyle=":", linewidth=1, label=f"chance (1/{P})")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_title(f"Hybrid p={P} m={NUM_KV_PAIRS} {n_layers}L - Accuracy")
    ax_acc.legend()

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "training.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {plot_path}")

    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------
    with open(os.path.join(out_dir, "status.txt"), "w") as f:
        f.write("COMPLETED\n")

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
