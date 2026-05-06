"""
scripts/run_hybrid_p23_m3.py

Hybrid retrieve-then-add with SMALL p=23, m=3 key-value pairs.

Rationale: p=113 with m=3 gives ~52M unique examples but only 11k training
examples (0.02% coverage). p=23 shrinks the combinatorial space dramatically:
  - Total unique examples: ~219k (23^3 * 6 query-pair combos * 6 permutations)
  - With 20k examples, training coverage ~9% — enough for grokking dynamics
  - d_vocab_out=23, chance=1/23=4.35%

Task: y = (v(q1) + v(q2)) mod 23
  p=23, num_kv_pairs=3, standard architecture, n_layers=3, seed=0

With m=3:
  n_ctx = 2*3 + 4 = 10
  d_vocab = 23 + 3 + 2 = 28
  d_vocab_out = 23

Sequence: [K0,V0, K1,V1, K2,V2, QUERY, q1, q2, EQ]

Outputs:
  outputs/hybrid_p23_m3_standard_seed0/
"""

from __future__ import annotations

import csv
import json
import os
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

from data.splits import get_hybrid_dataloaders_3way
from models.transformer import create_standard_transformer
from training.train_v2 import train_v2
from analysis.extract_activations import extract_activations
from analysis.compute_erank_profiles import compute_erank_profile

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED         = 0
P            = 23           # small prime for better data coverage
NUM_KV_PAIRS = 3
N_LAYERS     = 3
D_MODEL      = 128
N_HEADS      = 4
D_MLP        = 512
LR           = 0.0003
WEIGHT_DECAY = 1.0          # canonical for grokking
BATCH_SIZE   = 256
MAX_EPOCHS   = 50000
NUM_EXAMPLES = 20000
TRAIN_FRAC   = 0.7
FEASIBILITY_THRESHOLD = 0.10  # 10% (chance is ~4.35%)

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "hybrid_p23_m3_standard_seed0")


def label_sanity_check(train_loader, p, num_kv_pairs):
    """Verify labels for 20 examples. Returns (pass_bool, report_lines)."""
    lines = []
    lines.append("=" * 60)
    lines.append("LABEL SANITY CHECK — 20 examples")
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
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Hybrid p={P} m={NUM_KV_PAIRS} convergence experiment")

    with open(os.path.join(OUT_DIR, "status.txt"), "w") as f:
        f.write("RUNNING\n")

    n_ctx = 2 * NUM_KV_PAIRS + 4       # = 10
    d_vocab = P + NUM_KV_PAIRS + 2     # = 28
    d_vocab_out = P                     # = 23
    chance = 1.0 / P
    print(f"n_ctx={n_ctx} d_vocab={d_vocab} d_vocab_out={d_vocab_out} n_layers={N_LAYERS}")
    print(f"Chance accuracy: 1/{P} = {chance:.4f}")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print("Building dataloaders...")
    train_loader, val_loader, test_loader, split_meta = get_hybrid_dataloaders_3way(
        p=P,
        num_kv_pairs=NUM_KV_PAIRS,
        num_examples=NUM_EXAMPLES,
        train_frac=TRAIN_FRAC,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )
    print(f"  train={split_meta['n_train']} val={split_meta['n_val']} test={split_meta['n_test']}")

    # Compute data coverage
    total_space = (P ** NUM_KV_PAIRS) * (NUM_KV_PAIRS * (NUM_KV_PAIRS - 1))
    coverage = split_meta['n_train'] / total_space
    print(f"  Combinatorial space (value combos * query pairs): {total_space:,}")
    print(f"  Training coverage: {coverage:.2%}")

    split_path = os.path.join(OUT_DIR, "split_metadata.json")
    with open(split_path, "w") as f:
        json.dump(split_meta, f, indent=2)

    # -----------------------------------------------------------------------
    # Label sanity check
    # -----------------------------------------------------------------------
    print("\nRunning label sanity check...")
    sanity_pass, sanity_lines = label_sanity_check(train_loader, P, NUM_KV_PAIRS)
    sanity_text = "\n".join(sanity_lines)
    print(sanity_text)

    sanity_path = os.path.join(OUT_DIR, "label_sanity_check.txt")
    with open(sanity_path, "w") as f:
        f.write(sanity_text)

    if not sanity_pass:
        print("ERROR: Label sanity check FAILED. Aborting.")
        with open(os.path.join(OUT_DIR, "status.txt"), "w") as f:
            f.write("FAILED\n")
        sys.exit(1)

    print("Label sanity check PASSED. Proceeding to training.\n")

    # -----------------------------------------------------------------------
    # Model (3 layers)
    # -----------------------------------------------------------------------
    torch.manual_seed(SEED)
    model = create_standard_transformer(
        "hybrid_retrieve_add",
        cfg_overrides={
            "hybrid_p": P,
            "hybrid_num_kv_pairs": NUM_KV_PAIRS,
            "n_layers": N_LAYERS,
        },
        seed=SEED,
    )
    assert model.cfg.n_layers == N_LAYERS
    assert model.cfg.n_ctx == n_ctx
    assert model.cfg.d_vocab == d_vocab
    assert model.cfg.d_vocab_out == d_vocab_out
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: n_layers={model.cfg.n_layers} n_ctx={model.cfg.n_ctx} "
          f"d_model={model.cfg.d_model} n_heads={model.cfg.n_heads} "
          f"d_vocab={model.cfg.d_vocab} d_vocab_out={model.cfg.d_vocab_out}")
    print(f"Parameters: {n_params:,}")

    config_used = {
        "task": "hybrid_retrieve_add",
        "p": P,
        "num_kv_pairs": NUM_KV_PAIRS,
        "architecture": "standard",
        "n_layers": N_LAYERS,
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
        "num_examples": NUM_EXAMPLES,
        "train_frac": TRAIN_FRAC,
        "seed": SEED,
        "n_params": n_params,
        "combinatorial_space": total_space,
        "training_coverage": coverage,
    }
    with open(os.path.join(OUT_DIR, "config_used.yaml"), "w") as f:
        yaml.dump(config_used, f, default_flow_style=False)

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    ckpt_dir = os.path.join(OUT_DIR, "checkpoints")
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
    # Copy best-val checkpoint
    # -----------------------------------------------------------------------
    import shutil
    best_ckpt = os.path.join(ckpt_dir, "model_best_val.pt")
    best_val_ckpt_path = os.path.join(OUT_DIR, "best_val_checkpoint.pt")
    final_ckpt_path = os.path.join(OUT_DIR, "final_checkpoint.pt")

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
                    "n_layers": N_LAYERS,
                },
                seed=SEED,
            )
            model_for_erank.load_state_dict(ckpt_data["model_state_dict"])
            acts = extract_activations(
                model_for_erank, test_loader, "hybrid_retrieve_add",
                n_examples=500, device=device
            )
            erank_result = compute_erank_profile(acts, n_layers=N_LAYERS)

            da = erank_result.get("delta_erank_attn", {})
            dm = erank_result.get("delta_erank_mlp", {})
            sum_da = sum(da.values()) if da else 0.0
            sum_dm = sum(dm.values()) if dm else 0.0
            erank_result["sum_delta_attn"] = sum_da
            erank_result["sum_delta_mlp"] = sum_dm

            erank_csv_path = os.path.join(OUT_DIR, "endpoint_erank.csv")
            with open(erank_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["layer", "erank_pre", "erank_mid", "erank_post",
                                 "delta_attn", "delta_mlp"])
                pre = erank_result.get("erank_pre", {})
                mid = erank_result.get("erank_mid", {})
                post = erank_result.get("erank_post", {})
                for layer in range(N_LAYERS):
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

            erank_json_path = os.path.join(OUT_DIR, "endpoint_erank.json")
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
    _plot_training(history, os.path.join(OUT_DIR, "training.png"))

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    _write_report(
        out_dir=OUT_DIR,
        best_val_epoch=best_val_epoch,
        best_val_acc=best_val_acc,
        test_acc_at_bv=test_acc_at_bv,
        final_train_acc=final_train_acc,
        final_val_acc=final_val_acc,
        final_test_acc=final_test_acc,
        stopping_epoch=stopping_epoch,
        erank_result=erank_result,
        split_meta=split_meta,
        config_used=config_used,
        history=history,
        t_elapsed=t_elapsed,
    )

    with open(os.path.join(OUT_DIR, "status.txt"), "w") as f:
        f.write("COMPLETED\n")

    print(f"\nAll outputs saved to: {OUT_DIR}")


def _plot_training(history: dict, save_path: str):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = history["epoch"]

    if history.get("train_loss"):
        ax_loss.plot(epochs, history["train_loss"], label="train")
    if history.get("val_loss"):
        ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"Hybrid p={P} m={NUM_KV_PAIRS} — Loss")
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
    ax_acc.set_title(f"Hybrid p={P} m={NUM_KV_PAIRS} — Accuracy")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def _write_report(out_dir, best_val_epoch, best_val_acc, test_acc_at_bv,
                  final_train_acc, final_val_acc, final_test_acc,
                  stopping_epoch, erank_result, split_meta, config_used,
                  history, t_elapsed):
    lines = []
    a = lines.append
    chance = 1.0 / P

    a(f"# Hybrid p={P} m={NUM_KV_PAIRS} Convergence Report")
    a("")
    a("## 1. Executive summary")
    a("")
    if best_val_acc is not None and best_val_acc >= FEASIBILITY_THRESHOLD:
        a(f"The model **SOLVED or is learning** the hybrid task with p={P}.")
        a(f"Best val accuracy: {best_val_acc:.4f} (threshold: {FEASIBILITY_THRESHOLD})")
    elif best_val_acc is not None and best_val_acc > chance * 3:
        a(f"The model shows **PARTIAL LEARNING**.")
        a(f"Best val accuracy: {best_val_acc:.4f} — above chance ({chance:.4f}) but below threshold ({FEASIBILITY_THRESHOLD})")
    else:
        a(f"The model **FAILED** to generalise.")
        a(f"Best val accuracy: {best_val_acc:.4f} — near chance ({chance:.4f})")
    a("")

    a("## 2. Motivation")
    a("")
    a(f"p=113 with m=3 gave only 0.02% data coverage of the combinatorial space.")
    a(f"p={P} shrinks the space to ~{config_used.get('combinatorial_space', '?'):,} unique examples,")
    a(f"giving {config_used.get('training_coverage', 0):.1%} training coverage — enough for grokking.")
    a("")

    a("## 3. Run configuration")
    a("")
    for k, v in config_used.items():
        a(f"- {k}: {v}")
    a(f"- training time: {t_elapsed/3600:.2f} hours")
    a("")

    a("## 4. Training dynamics")
    a("")
    a("| Epoch | train_acc | val_acc | test_acc | train_loss | val_loss |")
    a("|---:|---:|---:|---:|---:|---:|")
    key_epochs = [0, 1, 10, 100, 500, 1000, 5000, 10000, 20000, best_val_epoch, stopping_epoch]
    key_epochs = sorted(set(e for e in key_epochs if e is not None))
    for ep in key_epochs:
        if ep in history["epoch"]:
            idx = history["epoch"].index(ep)
            ta = history["train_acc"][idx] if idx < len(history["train_acc"]) else 0
            va = history["val_acc"][idx] if idx < len(history["val_acc"]) else 0
            tea = history["test_acc"][idx] if idx < len(history["test_acc"]) else 0
            tl = history["train_loss"][idx] if idx < len(history["train_loss"]) else 0
            vl = history["val_loss"][idx] if idx < len(history["val_loss"]) else 0
            label = ""
            if ep == best_val_epoch:
                label = " *best val*"
            if ep == stopping_epoch:
                label += " *final*"
            a(f"| {ep}{label} | {ta:.4f} | {va:.4f} | {tea:.4f} | {tl:.4f} | {vl:.4f} |")
    a("")

    a("## 5. Best checkpoint and test result")
    a("")
    a(f"- Best val epoch: {best_val_epoch}")
    a(f"- Best val acc: {best_val_acc:.4f}")
    a(f"- Test acc @ best val: {test_acc_at_bv:.4f}" if test_acc_at_bv is not None else "- Test acc @ best val: N/A")
    a(f"- Final epoch: {stopping_epoch}")
    a(f"- Final train acc: {final_train_acc:.4f}" if final_train_acc is not None else "- Final train acc: N/A")
    a(f"- Final val acc: {final_val_acc:.4f}" if final_val_acc is not None else "- Final val acc: N/A")
    a(f"- Final test acc: {final_test_acc:.4f}" if final_test_acc is not None else "- Final test acc: N/A")
    a(f"- Chance: 1/{P} = {chance:.5f}")
    a("")

    a("## 6. Endpoint eRank")
    a("")
    if erank_result:
        da = erank_result.get("delta_erank_attn", {})
        dm = erank_result.get("delta_erank_mlp", {})
        pre = erank_result.get("erank_pre", {})
        mid = erank_result.get("erank_mid", {})
        post = erank_result.get("erank_post", {})
        sum_da = erank_result.get("sum_delta_attn", 0)
        sum_dm = erank_result.get("sum_delta_mlp", 0)

        a("| Layer | erank_pre | erank_mid | erank_post | delta_attn | delta_mlp |")
        a("|---:|---:|---:|---:|---:|---:|")
        for layer in range(N_LAYERS):
            lk = f"layer_{layer}"
            a(f"| {layer} | {pre.get(lk, 0):.3f} | {mid.get(lk, 0):.3f} | {post.get(lk, 0):.3f} | {da.get(lk, 0):.3f} | {dm.get(lk, 0):.3f} |")
        a("")
        a(f"- **sum_delta_attn**: {sum_da:.3f}")
        a(f"- **sum_delta_mlp**: {sum_dm:.3f}")
    else:
        a("eRank computation was not available.")
    a("")

    a("## 7. Comparison with p=113")
    a("")
    a("| Setting | p=113 m=3 | p=23 m=3 |")
    a("|---|---|---|")
    a(f"| Combinatorial space | ~52M | ~{config_used.get('combinatorial_space', '?'):,} |")
    a(f"| Training coverage | 0.02% | {config_used.get('training_coverage', 0):.1%} |")
    a(f"| Chance accuracy | 0.88% | {chance:.1%} |")
    a(f"| Best val accuracy | 3.75% (2L) / 4.18% (3L) | {best_val_acc:.2%} |")
    a(f"| Best test accuracy | 2.55% (2L) / 3.87% (3L) | {test_acc_at_bv:.2%}" if test_acc_at_bv else "| Best test accuracy | 2.55% / 3.87% | N/A |")

    report_path = os.path.join(out_dir, "hybrid_p23_m3_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
