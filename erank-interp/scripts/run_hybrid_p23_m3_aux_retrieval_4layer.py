"""
scripts/run_hybrid_p23_m3_aux_retrieval_4layer.py

Phase B rescue training for hybrid retrieve-then-add (p=23, m=3) with
auxiliary retrieval supervision.

The base model is the same 4-layer HookedTransformer used in the earlier
depth feasibility runs. We add two extra linear heads that consume the
final-layer residual stream at the EQ position and predict v(q1) and v(q2)
independently. The combined loss is:

    L = CE(sum_logits, y) + alpha * CE(r1_logits, r1) + alpha * CE(r2_logits, r2)

with alpha = 0.5. The retrieval heads are training aids and diagnostics;
final task success is judged by sum_val_acc / sum_test_acc.

Outputs are written to outputs/hybrid_p23_m3_aux_retrieval_4layer_seed0/.
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
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from data.splits import get_hybrid_dataloaders_3way_exhaustive
from models.transformer import create_standard_transformer

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

SEED = 0
P = 23
NUM_KV_PAIRS = 3
N_LAYERS = 4
D_MODEL = 128
N_HEADS = 4
D_MLP = 512
LR = 3e-4
WEIGHT_DECAY = 0.1
BATCH_SIZE = 256
MAX_EPOCHS = 10000
AUX_WEIGHT = 0.5

RUN_NAME = f"hybrid_p23_m3_aux_retrieval_{N_LAYERS}layer_seed{SEED}"


# ---------------------------------------------------------------------------
# Wrapper model with auxiliary heads
# ---------------------------------------------------------------------------

class HybridWithAuxHeads(nn.Module):
    """HookedTransformer base + two auxiliary linear heads (r1, r2) at EQ position."""

    def __init__(self, base_model, p: int):
        super().__init__()
        self.base = base_model
        self.p = p
        self.last_layer_idx = base_model.cfg.n_layers - 1
        self.resid_hook_name = f"blocks.{self.last_layer_idx}.hook_resid_post"
        self.aux_r1 = nn.Linear(base_model.cfg.d_model, p)
        self.aux_r2 = nn.Linear(base_model.cfg.d_model, p)

    def forward(self, tokens):
        logits, cache = self.base.run_with_cache(
            tokens, names_filter=self.resid_hook_name
        )
        resid_eq = cache[self.resid_hook_name][:, -1, :]  # (batch, d_model)
        sum_logits = logits[:, -1, :]
        r1_logits = self.aux_r1(resid_eq)
        r2_logits = self.aux_r2(resid_eq)
        return sum_logits, r1_logits, r2_logits


# ---------------------------------------------------------------------------
# Retrieval-label computation
# ---------------------------------------------------------------------------

def compute_retrieval_labels(inputs: torch.Tensor, p: int, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """For each example, return (r1, r2) = (v(q1_idx), v(q2_idx))."""
    key_offset = p
    kv_keys = inputs[:, 0:2 * m:2]      # (B, m) - key tokens
    kv_vals = inputs[:, 1:2 * m + 1:2]  # (B, m) - value tokens
    key_indices = kv_keys - key_offset  # (B, m), values in {0..m-1}
    q1_idx = inputs[:, 2 * m + 1] - key_offset  # (B,)
    q2_idx = inputs[:, 2 * m + 2] - key_offset  # (B,)
    match1 = (key_indices == q1_idx.unsqueeze(1)).long()
    match2 = (key_indices == q2_idx.unsqueeze(1)).long()
    r1 = (kv_vals * match1).sum(dim=1)
    r2 = (kv_vals * match2).sum(dim=1)
    return r1, r2


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def label_sanity_check(loader, p, m):
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("LABEL SANITY CHECK - 20 examples (sum + r1 + r2)")
    lines.append("=" * 60)
    inputs, sum_labels = next(iter(loader))
    r1, r2 = compute_retrieval_labels(inputs, p, m)
    n_check = min(20, inputs.shape[0])
    all_pass = True
    for i in range(n_check):
        seq = inputs[i].tolist()
        kv: dict[int, int] = {}
        for slot in range(m):
            k_idx = seq[2 * slot] - p
            v_val = seq[2 * slot + 1]
            kv[k_idx] = v_val
        q1_idx = seq[2 * m + 1] - p
        q2_idx = seq[2 * m + 2] - p
        expected_sum = (kv[q1_idx] + kv[q2_idx]) % p
        expected_r1 = kv[q1_idx]
        expected_r2 = kv[q2_idx]
        ok = (
            int(sum_labels[i]) == expected_sum
            and int(r1[i]) == expected_r1
            and int(r2[i]) == expected_r2
        )
        if not ok:
            all_pass = False
        lines.append(
            f"Ex {i+1:2d}: q1={q1_idx} q2={q2_idx} v(q1)={kv[q1_idx]} v(q2)={kv[q2_idx]} "
            f"-> sum={int(sum_labels[i])} (exp {expected_sum}), "
            f"r1={int(r1[i])} (exp {expected_r1}), "
            f"r2={int(r2[i])} (exp {expected_r2}) {'PASS' if ok else 'FAIL'}"
        )
    verdict = "ALL 20 PASSED" if all_pass else "SOME FAILED"
    lines.append("=" * 60)
    lines.append(f"VERDICT: {verdict}")
    lines.append("=" * 60)
    return all_pass, lines


# ---------------------------------------------------------------------------
# Eval and training loops
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: HybridWithAuxHeads, loader, device, p, m, criterion, alpha):
    model.eval()
    total_loss = 0.0
    n_total = 0
    n_sum_correct = 0
    n_r1_correct = 0
    n_r2_correct = 0
    for inputs, sum_labels in loader:
        inputs = inputs.to(device)
        sum_labels = sum_labels.to(device)
        r1_labels, r2_labels = compute_retrieval_labels(inputs, p, m)
        r1_labels = r1_labels.to(device)
        r2_labels = r2_labels.to(device)
        sum_logits, r1_logits, r2_logits = model(inputs)
        loss_sum = criterion(sum_logits, sum_labels)
        loss_r1 = criterion(r1_logits, r1_labels)
        loss_r2 = criterion(r2_logits, r2_labels)
        loss = loss_sum + alpha * loss_r1 + alpha * loss_r2
        total_loss += loss.item()
        n_sum_correct += (sum_logits.argmax(dim=-1) == sum_labels).sum().item()
        n_r1_correct += (r1_logits.argmax(dim=-1) == r1_labels).sum().item()
        n_r2_correct += (r2_logits.argmax(dim=-1) == r2_labels).sum().item()
        n_total += sum_labels.size(0)
    return {
        "loss": total_loss / max(len(loader), 1),
        "sum_acc": n_sum_correct / n_total,
        "r1_acc": n_r1_correct / n_total,
        "r2_acc": n_r2_correct / n_total,
    }


def train_one_epoch(model: HybridWithAuxHeads, loader, optimizer, device, p, m, criterion, alpha):
    model.train()
    total_loss = 0.0
    n_total = 0
    n_sum_correct = 0
    n_r1_correct = 0
    n_r2_correct = 0
    for inputs, sum_labels in loader:
        inputs = inputs.to(device)
        sum_labels = sum_labels.to(device)
        r1_labels, r2_labels = compute_retrieval_labels(inputs, p, m)
        r1_labels = r1_labels.to(device)
        r2_labels = r2_labels.to(device)
        sum_logits, r1_logits, r2_logits = model(inputs)
        loss_sum = criterion(sum_logits, sum_labels)
        loss_r1 = criterion(r1_logits, r1_labels)
        loss_r2 = criterion(r2_logits, r2_labels)
        loss = loss_sum + alpha * loss_r1 + alpha * loss_r2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_sum_correct += (sum_logits.argmax(dim=-1) == sum_labels).sum().item()
        n_r1_correct += (r1_logits.argmax(dim=-1) == r1_labels).sum().item()
        n_r2_correct += (r2_logits.argmax(dim=-1) == r2_labels).sum().item()
        n_total += sum_labels.size(0)
    return {
        "loss": total_loss / max(len(loader), 1),
        "sum_acc": n_sum_correct / n_total,
        "r1_acc": n_r1_correct / n_total,
        "r2_acc": n_r2_correct / n_total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n_layers", type=int, default=N_LAYERS)
    parser.add_argument("--aux_weight", type=float, default=AUX_WEIGHT)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    seed = args.seed
    n_layers = args.n_layers
    alpha = args.aux_weight
    lr = args.lr
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    max_epochs = args.max_epochs

    run_name = f"hybrid_p23_m3_aux_retrieval_{n_layers}layer_seed{seed}"
    out_dir = os.path.join(PROJECT_ROOT, "outputs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(out_dir, "logs")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(out_dir, "status.txt"), "w") as f:
        f.write("RUNNING\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(
        f"Phase B aux-retrieval rescue: p={P} m={NUM_KV_PAIRS} n_layers={n_layers} "
        f"seed={seed} alpha={alpha} lr={lr} wd={weight_decay} bs={batch_size}"
    )

    # Data ----------------------------------------------------------------
    train_loader, val_loader, test_loader, split_meta = get_hybrid_dataloaders_3way_exhaustive(
        p=P,
        num_kv_pairs=NUM_KV_PAIRS,
        train_frac=0.8,
        val_frac=0.1,
        batch_size=batch_size,
        seed=seed,
    )
    print(
        f"  total={split_meta['n_total']} train={split_meta['n_train']} "
        f"val={split_meta['n_val']} test={split_meta['n_test']}"
    )
    with open(os.path.join(out_dir, "split_metadata.json"), "w") as f:
        json.dump(split_meta, f, indent=2)

    # Sanity --------------------------------------------------------------
    sanity_pass, sanity_lines = label_sanity_check(train_loader, P, NUM_KV_PAIRS)
    sanity_text = "\n".join(sanity_lines)
    print(sanity_text)
    with open(os.path.join(out_dir, "label_sanity_check.txt"), "w") as f:
        f.write(sanity_text)
    if not sanity_pass:
        with open(os.path.join(out_dir, "status.txt"), "w") as f:
            f.write("FAILED\n")
        print("ERROR: Label sanity check FAILED. Aborting.")
        sys.exit(1)

    # Model ---------------------------------------------------------------
    torch.manual_seed(seed)
    base_model = create_standard_transformer(
        "hybrid_retrieve_add",
        cfg_overrides={
            "hybrid_p": P,
            "hybrid_num_kv_pairs": NUM_KV_PAIRS,
            "n_layers": n_layers,
        },
        seed=seed,
    )
    model = HybridWithAuxHeads(base_model, P).to(device)

    n_base_params = sum(p.numel() for p in base_model.parameters())
    n_aux_params = sum(p.numel() for p in model.aux_r1.parameters()) + sum(
        p.numel() for p in model.aux_r2.parameters()
    )
    print(f"Params: base={n_base_params:,}  aux={n_aux_params:,}")

    config_used = {
        "task": "hybrid_retrieve_add_small",
        "p": P,
        "num_kv_pairs": NUM_KV_PAIRS,
        "architecture": "standard",
        "n_layers": n_layers,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "d_mlp": D_MLP,
        "n_ctx": 2 * NUM_KV_PAIRS + 4,
        "d_vocab": P + NUM_KV_PAIRS + 2,
        "d_vocab_out": P,
        "optimizer": "AdamW",
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "aux_retrieval_weight": alpha,
        "dataset": "exhaustive",
        "n_total": split_meta["n_total"],
        "n_train": split_meta["n_train"],
        "n_val": split_meta["n_val"],
        "n_test": split_meta["n_test"],
        "train_frac": 0.8,
        "val_frac": 0.1,
        "seed": seed,
        "n_params_base": n_base_params,
        "n_params_aux": n_aux_params,
        "run_name": run_name,
        "phase": "B_aux_retrieval",
    }
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.dump(config_used, f, default_flow_style=False)

    # Optimizer + criterion ----------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Training loop -------------------------------------------------------
    history: dict = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "sum_train_acc": [],
        "sum_val_acc": [],
        "sum_test_acc": [],
        "r1_train_acc": [],
        "r1_val_acc": [],
        "r1_test_acc": [],
        "r2_train_acc": [],
        "r2_val_acc": [],
        "r2_test_acc": [],
        "best_sum_val_acc": 0.0,
        "best_val_epoch": None,
    }

    best_sum_val_acc = 0.0
    best_val_epoch = None
    consecutive_high = 0
    sustained_threshold_epochs = 200  # number of consecutive evals at >=0.95
    eval_every = 1  # full eval each epoch (dataset is small)
    best_ckpt_path = os.path.join(ckpt_dir, "model_best_val.pt")
    final_ckpt_path = os.path.join(ckpt_dir, "model_final.pt")

    t_start = time.time()
    for epoch in tqdm(range(1, max_epochs + 1), desc="train"):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, P, NUM_KV_PAIRS, criterion, alpha
        )

        if epoch % eval_every == 0:
            val_metrics = evaluate(model, val_loader, device, P, NUM_KV_PAIRS, criterion, alpha)
            test_metrics = evaluate(model, test_loader, device, P, NUM_KV_PAIRS, criterion, alpha)

            history["epoch"].append(epoch)
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["test_loss"].append(test_metrics["loss"])
            history["sum_train_acc"].append(train_metrics["sum_acc"])
            history["sum_val_acc"].append(val_metrics["sum_acc"])
            history["sum_test_acc"].append(test_metrics["sum_acc"])
            history["r1_train_acc"].append(train_metrics["r1_acc"])
            history["r1_val_acc"].append(val_metrics["r1_acc"])
            history["r1_test_acc"].append(test_metrics["r1_acc"])
            history["r2_train_acc"].append(train_metrics["r2_acc"])
            history["r2_val_acc"].append(val_metrics["r2_acc"])
            history["r2_test_acc"].append(test_metrics["r2_acc"])

            if epoch % 100 == 0 or epoch <= 10:
                print(
                    f"epoch {epoch:5d} | "
                    f"sum tr={train_metrics['sum_acc']:.3f} val={val_metrics['sum_acc']:.3f} "
                    f"test={test_metrics['sum_acc']:.3f} | "
                    f"r1 tr={train_metrics['r1_acc']:.3f} val={val_metrics['r1_acc']:.3f} | "
                    f"r2 tr={train_metrics['r2_acc']:.3f} val={val_metrics['r2_acc']:.3f}"
                )

            if val_metrics["sum_acc"] > best_sum_val_acc:
                best_sum_val_acc = val_metrics["sum_acc"]
                best_val_epoch = epoch
                history["best_sum_val_acc"] = best_sum_val_acc
                history["best_val_epoch"] = best_val_epoch
                torch.save(
                    {
                        "model_state_dict": model.base.state_dict(),
                        "aux_r1_state_dict": model.aux_r1.state_dict(),
                        "aux_r2_state_dict": model.aux_r2.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "task_name": "hybrid_retrieve_add",
                        "best_sum_val_acc": best_sum_val_acc,
                    },
                    best_ckpt_path,
                )

            if val_metrics["sum_acc"] >= 0.95:
                consecutive_high += 1
            else:
                consecutive_high = 0
            if consecutive_high >= sustained_threshold_epochs:
                print(f"\nEarly stop: sum_val_acc >= 0.95 for {sustained_threshold_epochs} epochs.")
                break

    elapsed = time.time() - t_start

    torch.save(
        {
            "model_state_dict": model.base.state_dict(),
            "aux_r1_state_dict": model.aux_r1.state_dict(),
            "aux_r2_state_dict": model.aux_r2.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": history["epoch"][-1] if history["epoch"] else 0,
            "task_name": "hybrid_retrieve_add",
        },
        final_ckpt_path,
    )

    print(
        f"\nTraining complete in {elapsed/3600:.2f} hours. "
        f"best epoch={best_val_epoch} best_sum_val_acc={best_sum_val_acc:.4f}"
    )

    # Save copies of checkpoints at top level for diagnostic compatibility
    if os.path.exists(best_ckpt_path):
        shutil.copy2(best_ckpt_path, os.path.join(out_dir, "best_val_checkpoint.pt"))
    shutil.copy2(final_ckpt_path, os.path.join(out_dir, "final_checkpoint.pt"))

    # Training log CSV
    log_csv = os.path.join(out_dir, "training_log.csv")
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "test_loss",
        "sum_train_acc",
        "sum_val_acc",
        "sum_test_acc",
        "r1_train_acc",
        "r1_val_acc",
        "r1_test_acc",
        "r2_train_acc",
        "r2_val_acc",
        "r2_test_acc",
        "best_sum_val_acc",
    ]
    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, ep in enumerate(history["epoch"]):
            writer.writerow(
                {
                    "epoch": ep,
                    "train_loss": history["train_loss"][i],
                    "val_loss": history["val_loss"][i],
                    "test_loss": history["test_loss"][i],
                    "sum_train_acc": history["sum_train_acc"][i],
                    "sum_val_acc": history["sum_val_acc"][i],
                    "sum_test_acc": history["sum_test_acc"][i],
                    "r1_train_acc": history["r1_train_acc"][i],
                    "r1_val_acc": history["r1_val_acc"][i],
                    "r1_test_acc": history["r1_test_acc"][i],
                    "r2_train_acc": history["r2_train_acc"][i],
                    "r2_val_acc": history["r2_val_acc"][i],
                    "r2_test_acc": history["r2_test_acc"][i],
                    "best_sum_val_acc": history["best_sum_val_acc"],
                }
            )
    print(f"Saved: {log_csv}")

    # Plots ---------------------------------------------------------------
    fig, (ax_loss, ax_sum, ax_aux) = plt.subplots(1, 3, figsize=(16, 4))
    epochs = history["epoch"]
    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Total loss")
    ax_loss.legend()

    ax_sum.plot(epochs, history["sum_train_acc"], label="train")
    ax_sum.plot(epochs, history["sum_val_acc"], label="val")
    ax_sum.plot(epochs, history["sum_test_acc"], label="test", linestyle="--", alpha=0.6)
    ax_sum.axhline(1 / P, color="red", linestyle=":", linewidth=1, label=f"chance (1/{P})")
    ax_sum.set_xlabel("Epoch")
    ax_sum.set_ylabel("sum accuracy")
    ax_sum.set_ylim(0, 1.05)
    ax_sum.set_title("sum_acc (task target)")
    ax_sum.legend()

    ax_aux.plot(epochs, history["r1_val_acc"], label="r1 val")
    ax_aux.plot(epochs, history["r2_val_acc"], label="r2 val")
    ax_aux.plot(epochs, history["r1_train_acc"], label="r1 train", alpha=0.5)
    ax_aux.plot(epochs, history["r2_train_acc"], label="r2 train", alpha=0.5)
    ax_aux.axhline(1 / P, color="red", linestyle=":", linewidth=1, label=f"chance (1/{P})")
    ax_aux.set_xlabel("Epoch")
    ax_aux.set_ylabel("retrieval accuracy")
    ax_aux.set_ylim(0, 1.05)
    ax_aux.set_title("Auxiliary heads (r1, r2)")
    ax_aux.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training.png"), dpi=150)
    plt.close(fig)

    # Best-checkpoint test results
    best_idx = (
        history["epoch"].index(best_val_epoch) if best_val_epoch in history["epoch"] else -1
    )
    best_sum_test = history["sum_test_acc"][best_idx] if best_idx >= 0 else None
    best_r1_test = history["r1_test_acc"][best_idx] if best_idx >= 0 else None
    best_r2_test = history["r2_test_acc"][best_idx] if best_idx >= 0 else None

    # Endpoint eRank (only if model meaningfully solved the task) ---------
    erank_done = False
    erank_sum_da = None
    erank_sum_dm = None
    erank_layer_rows: list[dict] = []
    try:
        from analysis.extract_activations import extract_activations
        from analysis.compute_erank_profiles import compute_erank_profile

        if best_sum_val_acc >= 0.40:  # we still compute it; clearly label as failed-model if low
            print("Computing endpoint eRank...")
            ckpt_data = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            er_model = create_standard_transformer(
                "hybrid_retrieve_add",
                cfg_overrides={
                    "hybrid_p": P,
                    "hybrid_num_kv_pairs": NUM_KV_PAIRS,
                    "n_layers": n_layers,
                },
                seed=seed,
            )
            er_model.load_state_dict(ckpt_data["model_state_dict"])
            er_model = er_model.to(device)
            acts = extract_activations(
                er_model,
                test_loader,
                "hybrid_retrieve_add",
                n_examples=500,
                device=device,
            )
            profile = compute_erank_profile(acts, n_layers=n_layers)
            da = profile.get("delta_erank_attn", {})
            dm = profile.get("delta_erank_mlp", {})
            erank_sum_da = sum(da.values()) if da else 0.0
            erank_sum_dm = sum(dm.values()) if dm else 0.0
            pre = profile.get("erank_pre", {})
            mid = profile.get("erank_mid", {})
            post = profile.get("erank_post", {})
            for layer in range(n_layers):
                lk = f"layer_{layer}"
                erank_layer_rows.append(
                    {
                        "layer": layer,
                        "erank_pre": pre.get(lk, ""),
                        "erank_mid": mid.get(lk, ""),
                        "erank_post": post.get(lk, ""),
                        "delta_attn": da.get(lk, ""),
                        "delta_mlp": dm.get(lk, ""),
                    }
                )
            erank_csv_path = os.path.join(out_dir, "endpoint_erank.csv")
            with open(erank_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    ["layer", "erank_pre", "erank_mid", "erank_post", "delta_attn", "delta_mlp"]
                )
                for row in erank_layer_rows:
                    w.writerow(
                        [
                            row["layer"],
                            row["erank_pre"],
                            row["erank_mid"],
                            row["erank_post"],
                            row["delta_attn"],
                            row["delta_mlp"],
                        ]
                    )
                w.writerow([])
                w.writerow(["sum_delta_attn", erank_sum_da])
                w.writerow(["sum_delta_mlp", erank_sum_dm])
            with open(os.path.join(out_dir, "endpoint_erank.json"), "w") as f:
                json.dump(profile, f, indent=2, default=float)
            erank_done = True
            print(f"  sum_delta_attn={erank_sum_da:.4f}  sum_delta_mlp={erank_sum_dm:.4f}")
        else:
            print(f"Skipping endpoint eRank: best_sum_val_acc={best_sum_val_acc:.3f} < 0.40 (failed model).")
    except Exception as e:
        print(f"  eRank computation failed: {e}")
        import traceback

        traceback.print_exc()

    # Re-run query diagnostics on the auxiliary-trained model -------------
    diag_dir = os.path.join(out_dir, "query_diagnostics_after_training")
    os.makedirs(os.path.join(diag_dir, "plots"), exist_ok=True)
    print("Running query diagnostics on the trained model...")
    try:
        from analysis import hybrid_query_diagnostics as hqd

        meta = {
            "name": run_name,
            "out_dir": out_dir,
            "ckpt_path": os.path.join(out_dir, "best_val_checkpoint.pt"),
            "ckpt_kind": "best_val",
            "config": config_used,
            "split_metadata": split_meta,
        }
        res = hqd.diagnose_model(meta, torch.device(device), diag_dir)

        # Write the standard CSV outputs
        hqd.write_csv(os.path.join(diag_dir, "diagnostics_all_examples.csv"), res["rows"])
        hqd.write_csv(os.path.join(diag_dir, "diagnostics_by_model.csv"), [res["summary"]])
        hqd.write_csv(os.path.join(diag_dir, "accuracy_by_query_pair.csv"), res["by_qpair"])
        hqd.write_csv(
            os.path.join(diag_dir, "candidate_match_rates.csv"),
            [{"model": run_name, "candidate": k, "match_rate": v} for k, v in res["match_rates"].items()],
        )
        qpairs = ["01", "02", "12"]
        col_names = ["pred_eq_s01", "pred_eq_s02", "pred_eq_s12", "pred_eq_none"]
        confusion_rows = []
        for ri, qp in enumerate(qpairs):
            row = {"model": run_name, "actual_query_pair": qp}
            for ci, cname in enumerate(col_names):
                row[cname] = float(res["confusion_norm"][ri, ci])
            confusion_rows.append(row)
        hqd.write_csv(os.path.join(diag_dir, "candidate_confusion_matrix.csv"), confusion_rows)
        hqd.write_csv(
            os.path.join(diag_dir, "counterfactual_query_results.csv"),
            res["counterfactual_rows"],
        )

        hqd.plot_accuracy_by_query_pair([res], os.path.join(diag_dir, "plots"))
        hqd.plot_match_rates([res], os.path.join(diag_dir, "plots"))
        hqd.plot_confusion([res], os.path.join(diag_dir, "plots"))
        hqd.plot_counterfactual([res], os.path.join(diag_dir, "plots"))
        hqd.write_report([res], diag_dir)
        diag_summary = res["summary"]
    except Exception as e:
        print(f"  Diagnostics on aux model failed: {e}")
        import traceback

        traceback.print_exc()
        diag_summary = None

    # Final report --------------------------------------------------------
    report_lines: list[str] = []
    report_lines.append("# Hybrid Auxiliary Retrieval Training Report\n")

    report_lines.append("## 1. Executive summary\n")
    if best_sum_val_acc >= 0.90:
        verdict = "Auxiliary retrieval supervision SOLVED the hybrid task."
    elif best_sum_val_acc >= 0.50:
        verdict = "Partial success — significantly above shortcut, but not fully solved."
    else:
        verdict = "Auxiliary retrieval supervision did NOT solve the hybrid task."
    report_lines.append(
        f"- best_sum_val_acc = **{best_sum_val_acc:.4f}** (epoch {best_val_epoch}); "
        f"sum_test@best = {best_sum_test if best_sum_test is None else f'{best_sum_test:.4f}'}; "
        f"r1_test@best = {best_r1_test if best_r1_test is None else f'{best_r1_test:.4f}'}; "
        f"r2_test@best = {best_r2_test if best_r2_test is None else f'{best_r2_test:.4f}'}"
    )
    report_lines.append(f"- {verdict}\n")

    report_lines.append("## 2. Training setup\n")
    report_lines.append(
        f"- run_name: {run_name}\n"
        f"- task: hybrid_retrieve_add (p={P}, m={NUM_KV_PAIRS})\n"
        f"- architecture: standard, n_layers={n_layers}, d_model={D_MODEL}, n_heads={N_HEADS}, d_mlp={D_MLP}\n"
        f"- optimizer: AdamW(lr={lr}, weight_decay={weight_decay}), batch_size={batch_size}\n"
        f"- aux_retrieval_weight (alpha): {alpha}\n"
        f"- max_epochs: {max_epochs}\n"
        f"- dataset: exhaustive 80/10/10 split, n_total={split_meta['n_total']}\n"
        f"- training time: {elapsed/3600:.2f} h\n"
    )

    report_lines.append("## 3. Label sanity check\n")
    report_lines.append("PASS\n" if sanity_pass else "FAIL\n")

    report_lines.append("## 4. Training dynamics\n")
    sample_epochs = [1, 100, 500, 1000, 2000, 4000, 6000, 8000, 10000]
    report_lines.append(
        "| epoch | sum_train | sum_val | r1_val | r2_val | train_loss | val_loss |"
    )
    report_lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for ep in sample_epochs:
        if ep in history["epoch"]:
            i = history["epoch"].index(ep)
            report_lines.append(
                f"| {ep} | {history['sum_train_acc'][i]:.3f} | {history['sum_val_acc'][i]:.3f} | "
                f"{history['r1_val_acc'][i]:.3f} | {history['r2_val_acc'][i]:.3f} | "
                f"{history['train_loss'][i]:.4f} | {history['val_loss'][i]:.4f} |"
            )
    if history["epoch"]:
        last_idx = -1
        last_ep = history["epoch"][last_idx]
        report_lines.append(
            f"| {last_ep} (last) | {history['sum_train_acc'][last_idx]:.3f} | "
            f"{history['sum_val_acc'][last_idx]:.3f} | {history['r1_val_acc'][last_idx]:.3f} | "
            f"{history['r2_val_acc'][last_idx]:.3f} | "
            f"{history['train_loss'][last_idx]:.4f} | {history['val_loss'][last_idx]:.4f} |"
        )
    report_lines.append("\n![training](training.png)\n")

    report_lines.append("## 5. Best checkpoint results\n")
    report_lines.append(
        f"- best_val_epoch: {best_val_epoch}\n"
        f"- sum_val_acc: {best_sum_val_acc:.4f}\n"
        f"- sum_test_acc: {best_sum_test:.4f}" if best_sum_test is not None else "- sum_test_acc: n/a"
    )
    report_lines.append(
        f"\n- r1_test_acc: {best_r1_test:.4f}" if best_r1_test is not None else "- r1_test_acc: n/a"
    )
    report_lines.append(
        f"- r2_test_acc: {best_r2_test:.4f}" if best_r2_test is not None else "- r2_test_acc: n/a"
    )
    report_lines.append("")

    report_lines.append("## 6. Query diagnostics after training\n")
    if diag_summary is not None:
        report_lines.append(
            f"- counterfactual_accuracy: {diag_summary['counterfactual_accuracy']:.3f}\n"
            f"- fraction_constant_prediction: {diag_summary['fraction_constant_prediction_across_queries']:.3f}\n"
            f"- verdict: **{diag_summary['verdict']}**"
        )
        report_lines.append(
            "\nSee `query_diagnostics_after_training/hybrid_query_diagnostics_report_for_chatgpt.md` for details."
        )
    else:
        report_lines.append("Diagnostics did not run (see logs).")
    report_lines.append("")

    report_lines.append("## 7. Endpoint eRank\n")
    if erank_done:
        report_lines.append(
            f"- sum_delta_attn: {erank_sum_da:.4f}\n- sum_delta_mlp: {erank_sum_dm:.4f}\n"
        )
        report_lines.append("| layer | pre | mid | post | Δ_attn | Δ_mlp |")
        report_lines.append("|---:|---:|---:|---:|---:|---:|")
        for r in erank_layer_rows:
            report_lines.append(
                f"| {r['layer']} | {r['erank_pre']:.3f} | {r['erank_mid']:.3f} | "
                f"{r['erank_post']:.3f} | {r['delta_attn']:.3f} | {r['delta_mlp']:.3f} |"
            )
        if best_sum_val_acc < 0.90:
            report_lines.append("\n**Note:** best_sum_val_acc below 0.90 — eRank reflects a partially-solved model.")
    else:
        report_lines.append("eRank not computed (model failed task threshold).")
    report_lines.append("")

    report_lines.append("## 8. Interpretation\n")
    if diag_summary is not None and diag_summary["verdict"] == "LIKELY_QUERY_CONDITIONED":
        report_lines.append("- Auxiliary retrieval supervision broke the fixed-pair shortcut: the model now uses the query.")
    elif diag_summary is not None and diag_summary["verdict"] == "LIKELY_FIXED_PAIR_SHORTCUT":
        report_lines.append("- Despite auxiliary supervision, the model still ignores the query at the sum head.")
    else:
        report_lines.append("- Diagnostics inconclusive; inspect retrieval-head accuracies vs. sum-head accuracy.")
    report_lines.append(
        f"- r1_val={history['r1_val_acc'][-1]:.3f}, r2_val={history['r2_val_acc'][-1]:.3f}: "
        "if these are high but sum_val is low, retrieval is solved internally but the addition is not."
    )
    report_lines.append("")

    report_lines.append("## 9. Recommended next action\n")
    if best_sum_val_acc >= 0.90:
        report_lines.append("- Run seeds 1, 2 to confirm reliability; consider attention-only control.")
    elif (
        history["r1_val_acc"]
        and history["r2_val_acc"]
        and history["r1_val_acc"][-1] >= 0.7
        and history["r2_val_acc"][-1] >= 0.7
        and best_sum_val_acc < 0.5
    ):
        report_lines.append("- Retrieval heads succeed but sum head does not. Tune MLP capacity / depth.")
    else:
        report_lines.append("- Retrieval heads also fail. Either simplify the task further (m=2) or stop the hybrid direction.")
    report_lines.append("")

    report_path = os.path.join(out_dir, "hybrid_aux_retrieval_report_for_chatgpt.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved: {report_path}")

    # Status --------------------------------------------------------------
    with open(os.path.join(out_dir, "status.txt"), "w") as f:
        f.write("COMPLETED\n")

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
