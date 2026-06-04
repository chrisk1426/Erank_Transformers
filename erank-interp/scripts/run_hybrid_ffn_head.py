"""
scripts/run_hybrid_ffn_head.py

Part B2 — hybrid retrieve-then-add (p=23, m=3) with a non-linear (FFN) output
head replacing the linear unembed.

Architecture: standard 4-layer HookedTransformer base. The transformer
predicts at the EQ (last) position. Instead of using its built-in linear
unembed, we feed the EQ-position residual into a 2-layer GELU MLP that
outputs logits over {0..p-1}.

Training loss is just CrossEntropy on sum_logits — no auxiliary supervision.
Outputs are written to outputs/hybrid_ffn_head/<run_name>/ with the same
file conventions as the existing hybrid scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from data.splits import get_hybrid_dataloaders_3way_exhaustive
from models.transformer import create_standard_transformer

# Defaults match the handoff doc.
SEED = 0
P = 23
NUM_KV_PAIRS = 3
N_LAYERS = 4
D_MODEL = 128
N_HEADS = 4
D_MLP = 512
LR = 3e-4
WEIGHT_DECAY = 0.1   # doc explicitly: lower than the aux script's 0.1, vs. earlier 1.0 runs
BATCH_SIZE = 256
MAX_EPOCHS = 10000
HEAD_HIDDEN_DIM = 256
HEAD_ACTIVATION = "gelu"
SUSTAINED_EARLY_STOP = 200   # consecutive eval-epochs at val_acc >= 0.95


class HybridWithFFNHead(nn.Module):
    """Standard hybrid HookedTransformer + FFN output head consuming EQ residual."""

    def __init__(self, base_model, p: int, hidden_dim: int = HEAD_HIDDEN_DIM):
        super().__init__()
        self.base = base_model
        self.p = p
        self.last_layer_idx = base_model.cfg.n_layers - 1
        self.resid_hook_name = f"blocks.{self.last_layer_idx}.hook_resid_post"
        self.head_fc1 = nn.Linear(base_model.cfg.d_model, hidden_dim)
        self.head_fc2 = nn.Linear(hidden_dim, p)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _logits, cache = self.base.run_with_cache(
            tokens, names_filter=self.resid_hook_name
        )
        z_eq = cache[self.resid_hook_name][:, -1, :]  # (B, d_model)
        h = F.gelu(self.head_fc1(z_eq))
        return self.head_fc2(h)  # (B, p)


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> dict:
    model.eval()
    total_loss = 0.0; n = 0; n_correct = 0
    for inputs, labels in loader:
        inputs = inputs.to(device); labels = labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_correct += (logits.argmax(dim=-1) == labels).sum().item()
        n += labels.size(0)
    return {"loss": total_loss / max(len(loader), 1), "acc": n_correct / n}


def train_one_epoch(model, loader, optimizer, device, criterion) -> dict:
    model.train()
    total_loss = 0.0; n = 0; n_correct = 0
    for inputs, labels in loader:
        inputs = inputs.to(device); labels = labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
        n_correct += (logits.argmax(dim=-1) == labels).sum().item()
        n += labels.size(0)
    return {"loss": total_loss / max(len(loader), 1), "acc": n_correct / n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n_layers", type=int, default=N_LAYERS)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--head_hidden_dim", type=int, default=HEAD_HIDDEN_DIM)
    parser.add_argument("--p", type=int, default=P)
    parser.add_argument("--num_kv_pairs", type=int, default=NUM_KV_PAIRS)
    parser.add_argument("--output_root", default="outputs/hybrid_ffn_head")
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    seed = args.seed
    n_layers = args.n_layers
    p = args.p
    m = args.num_kv_pairs
    run_name = args.run_name or f"hybrid_p{p}_m{m}_{n_layers}layer_ffn_head_seed{seed}"
    out_dir = os.path.join(PROJECT_ROOT, args.output_root, run_name)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(out_dir, "status.txt"), "w") as f:
        f.write("RUNNING\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{run_name}] device={device}")
    print(f"[{run_name}] cfg: p={p} m={m} n_layers={n_layers} lr={args.lr} wd={args.weight_decay} "
          f"bs={args.batch_size} head_hidden={args.head_hidden_dim} max_epochs={args.max_epochs}")

    train_loader, val_loader, test_loader, split_meta = get_hybrid_dataloaders_3way_exhaustive(
        p=p, num_kv_pairs=m, train_frac=0.8, val_frac=0.1,
        batch_size=args.batch_size, seed=seed,
    )
    with open(os.path.join(out_dir, "split_metadata.json"), "w") as f:
        json.dump(split_meta, f, indent=2)
    print(f"[{run_name}] split: {split_meta}")

    torch.manual_seed(seed)
    base = create_standard_transformer(
        "hybrid_retrieve_add",
        cfg_overrides={
            "hybrid_p": p, "hybrid_num_kv_pairs": m, "n_layers": n_layers,
        },
        seed=seed,
    )
    model = HybridWithFFNHead(base, p, hidden_dim=args.head_hidden_dim).to(device)

    n_base = sum(t.numel() for t in base.parameters())
    n_head = sum(t.numel() for t in model.head_fc1.parameters()) + \
             sum(t.numel() for t in model.head_fc2.parameters())
    print(f"[{run_name}] params: base={n_base:,}  head={n_head:,}")

    config_used = {
        "task": "hybrid_retrieve_add_small",
        "p": p, "num_kv_pairs": m,
        "architecture": "standard",
        "n_layers": n_layers,
        "d_model": D_MODEL, "n_heads": N_HEADS, "d_mlp": D_MLP,
        "n_ctx": 2 * m + 4,
        "d_vocab": p + m + 2,
        "d_vocab_out": p,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "output_head_type": "ffn",
        "output_head_hidden_dim": args.head_hidden_dim,
        "output_head_activation": HEAD_ACTIVATION,
        "n_total": split_meta["n_total"], "n_train": split_meta["n_train"],
        "n_val": split_meta["n_val"], "n_test": split_meta["n_test"],
        "train_frac": 0.8, "val_frac": 0.1,
        "seed": seed,
        "n_params_base": n_base, "n_params_head": n_head,
        "run_name": run_name,
        "phase": "B2_ffn_head",
    }
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.dump(config_used, f, default_flow_style=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history: dict = {
        "epoch": [], "train_loss": [], "val_loss": [], "test_loss": [],
        "train_acc": [], "val_acc": [], "test_acc": [],
    }
    best_val_acc = -1.0; best_val_epoch = None; sustained = 0
    best_ckpt = os.path.join(out_dir, "best_val_checkpoint.pt")
    final_ckpt = os.path.join(out_dir, "final_checkpoint.pt")

    t_start = time.time()
    for epoch in tqdm(range(1, args.max_epochs + 1), desc=f"[{run_name}]"):
        tr = train_one_epoch(model, train_loader, optimizer, device, criterion)
        va = evaluate(model, val_loader, device, criterion)
        te = evaluate(model, test_loader, device, criterion)
        history["epoch"].append(epoch)
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["test_loss"].append(te["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_acc"].append(va["acc"])
        history["test_acc"].append(te["acc"])

        if epoch % 100 == 0 or epoch <= 5:
            print(f"  ep {epoch:5d} | tr {tr['acc']:.3f} val {va['acc']:.3f} test {te['acc']:.3f} | "
                  f"tr_loss {tr['loss']:.4f} val_loss {va['loss']:.4f}")

        if va["acc"] > best_val_acc:
            best_val_acc = va["acc"]; best_val_epoch = epoch
            torch.save({
                "base_state_dict": model.base.state_dict(),
                "head_fc1_state_dict": model.head_fc1.state_dict(),
                "head_fc2_state_dict": model.head_fc2.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": best_val_acc,
            }, best_ckpt)

        sustained = sustained + 1 if va["acc"] >= 0.95 else 0
        if sustained >= SUSTAINED_EARLY_STOP:
            print(f"\n[{run_name}] Early stop at ep {epoch} — val_acc >= 0.95 for {sustained} consec.")
            break

    elapsed = time.time() - t_start

    torch.save({
        "base_state_dict": model.base.state_dict(),
        "head_fc1_state_dict": model.head_fc1.state_dict(),
        "head_fc2_state_dict": model.head_fc2.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": history["epoch"][-1] if history["epoch"] else 0,
    }, final_ckpt)

    # CSV log
    with open(os.path.join(out_dir, "training_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "test_loss",
                    "train_acc", "val_acc", "test_acc"])
        for i, ep in enumerate(history["epoch"]):
            w.writerow([
                ep,
                history["train_loss"][i], history["val_loss"][i], history["test_loss"][i],
                history["train_acc"][i], history["val_acc"][i], history["test_acc"][i],
            ])

    # Plot
    try:
        eps = history["epoch"]
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
        ax_loss.plot(eps, history["train_loss"], label="train")
        ax_loss.plot(eps, history["val_loss"], label="val")
        ax_loss.plot(eps, history["test_loss"], label="test", linestyle="--", alpha=0.6)
        ax_loss.set_yscale("log")
        ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss"); ax_loss.legend()
        ax_loss.set_title(f"{run_name}  loss")
        ax_acc.plot(eps, history["train_acc"], label="train")
        ax_acc.plot(eps, history["val_acc"], label="val")
        ax_acc.plot(eps, history["test_acc"], label="test", linestyle="--", alpha=0.6)
        ax_acc.axhline(1 / p, color="red", linestyle=":", linewidth=1, label=f"chance (1/{p})")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title(f"{run_name}  accuracy")
        ax_acc.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "training.png"), dpi=140)
        plt.close(fig)
    except Exception:
        traceback.print_exc()

    # Best-checkpoint metrics
    test_at_best = None
    if best_val_epoch is not None and best_val_epoch in history["epoch"]:
        i = history["epoch"].index(best_val_epoch)
        test_at_best = history["test_acc"][i]

    # Endpoint eRank — only if model meaningfully solved
    erank_done = False
    sum_da = sum_dm = None
    erank_layer_rows: list[dict] = []
    try:
        from analysis.compute_erank_profiles import compute_erank_profile
        from analysis.extract_activations import extract_activations
        if best_val_acc >= 0.40:
            ckpt_data = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            er_base = create_standard_transformer(
                "hybrid_retrieve_add",
                cfg_overrides={
                    "hybrid_p": p, "hybrid_num_kv_pairs": m, "n_layers": n_layers,
                },
                seed=seed,
            )
            er_base.load_state_dict(ckpt_data["base_state_dict"])
            er_base = er_base.to(device)
            acts = extract_activations(
                er_base, test_loader, "hybrid_retrieve_add",
                n_examples=500, device=str(device),
            )
            profile = compute_erank_profile(acts, n_layers=n_layers)
            da = profile["delta_erank_attn"]; dm = profile["delta_erank_mlp"]
            sum_da = float(sum(da.values())); sum_dm = float(sum(dm.values()))
            for li in range(n_layers):
                lk = f"layer_{li}"
                erank_layer_rows.append({
                    "layer": li,
                    "erank_pre":  profile["erank"][lk]["pre"],
                    "erank_mid":  profile["erank"][lk]["mid"],
                    "erank_post": profile["erank"][lk]["post"],
                    "delta_attn": da[lk],
                    "delta_mlp":  dm[lk],
                })
            with open(os.path.join(out_dir, "endpoint_erank.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["layer", "erank_pre", "erank_mid", "erank_post", "delta_attn", "delta_mlp"])
                for r in erank_layer_rows:
                    w.writerow([r["layer"], r["erank_pre"], r["erank_mid"], r["erank_post"],
                                r["delta_attn"], r["delta_mlp"]])
                w.writerow([])
                w.writerow(["sum_delta_attn", sum_da])
                w.writerow(["sum_delta_mlp", sum_dm])
            with open(os.path.join(out_dir, "endpoint_erank.json"), "w") as f:
                json.dump(profile, f, indent=2, default=float)
            erank_done = True
        else:
            print(f"[{run_name}] Skipping endpoint eRank (best_val_acc={best_val_acc:.3f} < 0.40)")
    except Exception:
        print(f"[{run_name}] endpoint eRank failed:")
        traceback.print_exc()

    # Run report
    lines: list[str] = []
    lines.append(f"# Hybrid FFN-Head Run: {run_name}")
    lines.append("")
    lines.append("## 1. Setup")
    lines.append(f"- task: hybrid_retrieve_add (p={p}, m={m})")
    lines.append(f"- architecture: standard, n_layers={n_layers}, d_model={D_MODEL}, n_heads={N_HEADS}")
    lines.append(f"- output head: FFN, hidden={args.head_hidden_dim}, activation={HEAD_ACTIVATION}")
    lines.append(f"- optimizer: AdamW(lr={args.lr}, wd={args.weight_decay}), batch={args.batch_size}")
    lines.append(f"- training time: {elapsed/3600:.2f} h, ran for {history['epoch'][-1] if history['epoch'] else 0} epochs")
    lines.append("")
    lines.append("## 2. Best checkpoint")
    lines.append(f"- best_val_acc = {best_val_acc:.4f} @ epoch {best_val_epoch}")
    lines.append(f"- test_acc @ best-val = {test_at_best if test_at_best is None else f'{test_at_best:.4f}'}")
    lines.append("")
    lines.append("## 3. Endpoint eRank")
    if erank_done:
        lines.append(f"- sum_delta_attn = {sum_da:.4f}, sum_delta_mlp = {sum_dm:.4f}")
        lines.append("")
        lines.append("| layer | pre | mid | post | Δ_attn | Δ_mlp |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for r in erank_layer_rows:
            lines.append(
                f"| {r['layer']} | {r['erank_pre']:.3f} | {r['erank_mid']:.3f} | "
                f"{r['erank_post']:.3f} | {r['delta_attn']:+.3f} | {r['delta_mlp']:+.3f} |"
            )
    else:
        lines.append("Not computed (model did not meaningfully solve, or extraction failed).")
    lines.append("")
    lines.append("![training](training.png)")
    with open(os.path.join(out_dir, "run_report_for_chatgpt.md"), "w") as f:
        f.write("\n".join(lines))

    with open(os.path.join(out_dir, "status.txt"), "w") as f:
        f.write("COMPLETED\n")
    print(f"[{run_name}] DONE. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
