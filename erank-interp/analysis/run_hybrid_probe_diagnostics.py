"""
analysis/run_hybrid_probe_diagnostics.py

Frozen-probe diagnostics for hybrid retrieve-then-add (p=23, m=3).

For each available trained hybrid p=23,m=3 checkpoint:
  - Load model and reconstruct dataloaders.
  - Extract final-layer residual stream at the EQ (last) position.
  - Train two probe types:
        linear:  Wz + b
        mlp:     W2 * gelu(W1 z + b1) + b2  (hidden = 256)
  - Train probes for three targets:
        sum: y = (v(q1) + v(q2)) % p
        r1:  v(q1)
        r2:  v(q2)

Probes are trained on the model's *training* split features (the same examples
the transformer saw), evaluated on val + test features. The transformer is
never updated.

Outputs (under outputs/hybrid_probe_diagnostics/):
    probe_all_results.csv
    probe_summary.csv
    plots/probe_accuracy_by_target.png
    plots/linear_vs_mlp_probe_accuracy.png
    plots/retrieval_vs_sum_probe_accuracy.png
    hybrid_probe_diagnostics_report_for_chatgpt.md
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

from data.splits import get_hybrid_dataloaders_3way_exhaustive
from models.transformer import create_standard_transformer

# ---------------------------------------------------------------------------
# Candidate checkpoints to probe
# ---------------------------------------------------------------------------

CANDIDATE_DIRS = [
    "outputs/hybrid_small_p23_m3_standard_2layer_seed0",
    "outputs/hybrid_small_p23_m3_standard_3layer_seed0",
    "outputs/hybrid_small_p23_m3_standard_4layer_seed0",
    # Older p=23 hybrid_p23 directory (if present and distinct).
    "outputs/hybrid_p23",
    "outputs/hybrid_small_p23_3layer",
    "outputs/hybrid_small_p23_4layer",
]

PROBE_HIDDEN_DIM = 256
DEFAULT_PROBE_EPOCHS = 300
PROBE_LR = 3e-3
PROBE_WD = 1e-3
PROBE_BATCH = 512

# Mutated from CLI in main(); read by train_probe.
_PROBE_EPOCHS = DEFAULT_PROBE_EPOCHS


# ---------------------------------------------------------------------------
# Retrieval-label helper (mirrors run_hybrid_p23_m3_aux_retrieval_4layer.py)
# ---------------------------------------------------------------------------

def compute_retrieval_labels(
    inputs: torch.Tensor, p: int, m: int
) -> tuple[torch.Tensor, torch.Tensor]:
    key_offset = p
    kv_keys = inputs[:, 0:2 * m:2]
    kv_vals = inputs[:, 1:2 * m + 1:2]
    key_indices = kv_keys - key_offset
    q1_idx = inputs[:, 2 * m + 1] - key_offset
    q2_idx = inputs[:, 2 * m + 2] - key_offset
    match1 = (key_indices == q1_idx.unsqueeze(1)).long()
    match2 = (key_indices == q2_idx.unsqueeze(1)).long()
    r1 = (kv_vals * match1).sum(dim=1)
    r2 = (kv_vals * match2).sum(dim=1)
    return r1, r2


# ---------------------------------------------------------------------------
# Probe definitions
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_in, n_classes)

    def forward(self, x): return self.fc(x)


class MLPProbe(nn.Module):
    def __init__(self, d_in: int, hidden: int, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_eq_resid(
    model, loader: DataLoader, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (z, inputs_concat) — z has shape (N, d_model), inputs has shape (N, n_ctx).
    z is the final-layer residual stream at the EQ (last) position.
    """
    model.eval()
    n_layers = model.cfg.n_layers
    hook = f"blocks.{n_layers - 1}.hook_resid_post"
    zs: list[torch.Tensor] = []
    xs: list[torch.Tensor] = []
    for inputs, _labels in loader:
        inputs = inputs.to(device)
        _logits, cache = model.run_with_cache(inputs, names_filter=hook)
        z = cache[hook][:, -1, :].detach().cpu()
        zs.append(z)
        xs.append(inputs.detach().cpu())
    return torch.cat(zs, dim=0), torch.cat(xs, dim=0)


# ---------------------------------------------------------------------------
# Probe training (single target)
# ---------------------------------------------------------------------------

def train_probe(
    probe: nn.Module,
    z_train: torch.Tensor, y_train: torch.Tensor,
    z_val: torch.Tensor, y_val: torch.Tensor,
    z_test: torch.Tensor, y_test: torch.Tensor,
    device: torch.device,
) -> dict:
    probe = probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD)
    crit = nn.CrossEntropyLoss()

    train_ds = TensorDataset(z_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=PROBE_BATCH, shuffle=True)

    best_val_acc = -1.0
    best_state: dict | None = None
    best_train_acc = 0.0
    best_train_loss = float("inf")

    for ep in range(1, _PROBE_EPOCHS + 1):
        probe.train()
        ep_loss = 0.0
        n_correct = 0
        n_total = 0
        for zb, yb in train_loader:
            zb = zb.to(device); yb = yb.to(device)
            logits = probe(zb)
            loss = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * yb.size(0)
            n_correct += (logits.argmax(dim=-1) == yb).sum().item()
            n_total += yb.size(0)
        train_loss = ep_loss / n_total
        train_acc = n_correct / n_total

        probe.eval()
        with torch.no_grad():
            v_logits = probe(z_val.to(device))
            v_loss = crit(v_logits, y_val.to(device)).item()
            v_acc = (v_logits.argmax(dim=-1).cpu() == y_val).float().mean().item()
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state = {k: v.detach().clone().cpu() for k, v in probe.state_dict().items()}
            best_train_acc = train_acc
            best_train_loss = train_loss

    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        t_logits = probe(z_test.to(device))
        t_loss = crit(t_logits, y_test.to(device)).item()
        t_acc = (t_logits.argmax(dim=-1).cpu() == y_test).float().mean().item()
        v_logits = probe(z_val.to(device))
        v_loss = crit(v_logits, y_val.to(device)).item()
        v_acc = (v_logits.argmax(dim=-1).cpu() == y_val).float().mean().item()

    return {
        "train_acc": best_train_acc,
        "train_loss": best_train_loss,
        "val_acc": v_acc,
        "val_loss": v_loss,
        "test_acc": t_acc,
        "test_loss": t_loss,
    }


# ---------------------------------------------------------------------------
# Per-checkpoint orchestration
# ---------------------------------------------------------------------------

def model_dir_signature(d: str) -> str:
    """A stable name for dedup keys."""
    return os.path.normpath(d)


def load_model_for_dir(out_dir: str, device: torch.device):
    """
    Returns (model, cfg, ckpt_path, ckpt_type) or None if checkpoint missing.

    For aux-retrieval-trained checkpoints, model_state_dict carries the *base*
    HookedTransformer weights. For standard checkpoints, same convention.
    """
    cfg_path = os.path.join(out_dir, "config_used.yaml")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    p = int(cfg.get("p", cfg.get("hybrid_p", 23)))
    m = int(cfg.get("num_kv_pairs", cfg.get("hybrid_num_kv_pairs", 3)))
    n_layers = int(cfg.get("n_layers", 4))
    seed = int(cfg.get("seed", 0))

    best = os.path.join(out_dir, "best_val_checkpoint.pt")
    final = os.path.join(out_dir, "final_checkpoint.pt")
    if os.path.exists(best):
        ckpt_path, ckpt_type = best, "best_val"
    elif os.path.exists(final):
        ckpt_path, ckpt_type = final, "final"
    else:
        return None
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in state:
        return None

    model = create_standard_transformer(
        "hybrid_retrieve_add",
        cfg_overrides={
            "hybrid_p": p, "hybrid_num_kv_pairs": m, "n_layers": n_layers,
        },
        seed=seed,
    )
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, {"p": p, "m": m, "n_layers": n_layers, "seed": seed}, ckpt_path, ckpt_type


def diagnose_one(model, meta: dict, device: torch.device, run_label: str) -> list[dict]:
    """Run all 6 probes (3 targets × 2 types) for one model. Return rows for CSV."""
    p = meta["p"]; m = meta["m"]; n_layers = meta["n_layers"]; seed = meta["seed"]
    train_loader, val_loader, test_loader, split_meta = get_hybrid_dataloaders_3way_exhaustive(
        p=p, num_kv_pairs=m, train_frac=0.8, val_frac=0.1, batch_size=512, seed=seed,
    )
    print(f"  [{run_label}] split: total={split_meta['n_total']} "
          f"train={split_meta['n_train']} val={split_meta['n_val']} test={split_meta['n_test']}")

    # Extract activations + inputs for all three splits.
    z_tr, x_tr = extract_eq_resid(model, train_loader, device)
    z_va, x_va = extract_eq_resid(model, val_loader, device)
    z_te, x_te = extract_eq_resid(model, test_loader, device)

    # Build all 3 target label tensors.
    sum_tr = (x_tr[:, -1] + 0).long()  # placeholder; we'll recompute below
    # Pull labels directly from the loaders' tensor datasets to avoid
    # recomputing sum from inputs (more robust).
    y_sum_tr = train_loader.dataset.tensors[1].long()
    y_sum_va = val_loader.dataset.tensors[1].long()
    y_sum_te = test_loader.dataset.tensors[1].long()

    # NOTE: the dataloaders shuffle train; extract_eq_resid iterates them in
    # order. For the ORDER to align z_tr <-> y_sum_tr we need to run the train
    # loader without shuffling. Re-extract with a non-shuffled wrapper.
    train_unshuf = DataLoader(train_loader.dataset, batch_size=512, shuffle=False)
    z_tr, x_tr = extract_eq_resid(model, train_unshuf, device)

    r1_tr, r2_tr = compute_retrieval_labels(x_tr, p, m)
    r1_va, r2_va = compute_retrieval_labels(x_va, p, m)
    r1_te, r2_te = compute_retrieval_labels(x_te, p, m)

    targets = {
        "sum": (y_sum_tr, y_sum_va, y_sum_te),
        "r1":  (r1_tr.long(), r1_va.long(), r1_te.long()),
        "r2":  (r2_tr.long(), r2_va.long(), r2_te.long()),
    }
    rows: list[dict] = []
    d_in = z_tr.shape[1]
    for target_name, (yt, yv, yte) in targets.items():
        for probe_type in ("linear", "mlp"):
            t0 = time.time()
            probe = (
                LinearProbe(d_in, p) if probe_type == "linear"
                else MLPProbe(d_in, PROBE_HIDDEN_DIM, p)
            )
            res = train_probe(probe, z_tr, yt, z_va, yv, z_te, yte, device)
            elapsed = time.time() - t0
            rows.append({
                "model_name": run_label,
                "p": p, "m": m, "layers": n_layers, "seed": seed,
                "probe_type": probe_type,
                "target": target_name,
                "train_acc": res["train_acc"], "train_loss": res["train_loss"],
                "val_acc": res["val_acc"],     "val_loss": res["val_loss"],
                "test_acc": res["test_acc"],   "test_loss": res["test_loss"],
                "probe_hidden_dim_if_mlp": (PROBE_HIDDEN_DIM if probe_type == "mlp" else ""),
                "elapsed_sec": f"{elapsed:.1f}",
                "status": "OK",
            })
            print(
                f"    [{run_label}] {probe_type:>6} probe / {target_name:>3}: "
                f"train={res['train_acc']:.3f} val={res['val_acc']:.3f} test={res['test_acc']:.3f} "
                f"({elapsed:.1f}s)"
            )
    return rows


# ---------------------------------------------------------------------------
# Aggregation + plots
# ---------------------------------------------------------------------------

def write_results(rows: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "plots"); os.makedirs(plot_dir, exist_ok=True)
    fields = [
        "model_name", "checkpoint_path", "checkpoint_type",
        "layers", "p", "m", "seed",
        "probe_type", "target",
        "train_acc", "val_acc", "test_acc",
        "train_loss", "val_loss", "test_loss",
        "probe_hidden_dim_if_mlp", "elapsed_sec", "status",
    ]
    # Fill missing keys for CSV compatibility.
    full_rows: list[dict] = []
    for r in rows:
        out = {k: r.get(k, "") for k in fields}
        full_rows.append(out)
    p1 = os.path.join(out_dir, "probe_all_results.csv")
    with open(p1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in full_rows:
            w.writerow(r)
    print(f"  wrote {p1}")

    # Summary: one row per (model, target, probe_type) — same as full but useful pivot.
    p2 = os.path.join(out_dir, "probe_summary.csv")
    with open(p2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_name", "layers", "target", "probe_type",
                    "train_acc", "val_acc", "test_acc"])
        for r in rows:
            w.writerow([
                r["model_name"], r["layers"], r["target"], r["probe_type"],
                f"{r['train_acc']:.4f}", f"{r['val_acc']:.4f}", f"{r['test_acc']:.4f}",
            ])
    print(f"  wrote {p2}")


def make_plots(rows: list[dict], out_dir: str) -> None:
    plot_dir = os.path.join(out_dir, "plots"); os.makedirs(plot_dir, exist_ok=True)
    if not rows:
        return
    models = sorted({r["model_name"] for r in rows})
    targets = ["sum", "r1", "r2"]
    types = ["linear", "mlp"]

    # Plot 1: accuracy by target (one bar group per model, color = target, shape via probe type)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, ptype in zip(axes, types):
        x = np.arange(len(models))
        width = 0.25
        for ti, t in enumerate(targets):
            vals = [
                next((r["test_acc"] for r in rows
                      if r["model_name"] == m and r["target"] == t and r["probe_type"] == ptype), 0.0)
                for m in models
            ]
            ax.bar(x + (ti - 1) * width, vals, width, label=t)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax.axhline(1 / 23, color="red", linestyle=":", linewidth=1, label="chance (1/23)")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("test accuracy")
        ax.set_title(f"{ptype} probe — test acc by target")
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(plot_dir, "probe_accuracy_by_target.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"  wrote {p}")

    # Plot 2: linear vs MLP, faceted by target
    fig, axes = plt.subplots(1, len(targets), figsize=(4 * len(targets), 4), sharey=True)
    for ax, t in zip(axes, targets):
        x = np.arange(len(models))
        width = 0.4
        lin = [next((r["test_acc"] for r in rows
                     if r["model_name"] == m and r["target"] == t and r["probe_type"] == "linear"), 0.0)
               for m in models]
        mlp = [next((r["test_acc"] for r in rows
                     if r["model_name"] == m and r["target"] == t and r["probe_type"] == "mlp"), 0.0)
               for m in models]
        ax.bar(x - width / 2, lin, width, label="linear", color="steelblue")
        ax.bar(x + width / 2, mlp, width, label="mlp",    color="darkorange")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(1 / 23, color="red", linestyle=":", linewidth=1)
        ax.set_title(f"target = {t}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("test accuracy")
    fig.suptitle("Linear vs MLP probe accuracy")
    fig.tight_layout()
    p = os.path.join(plot_dir, "linear_vs_mlp_probe_accuracy.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"  wrote {p}")

    # Plot 3: retrieval (r1+r2 mean) vs sum
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.4
    x = np.arange(len(models))
    sum_acc = [
        next((r["test_acc"] for r in rows
              if r["model_name"] == m and r["target"] == "sum" and r["probe_type"] == "mlp"), 0.0)
        for m in models
    ]
    ret_acc = []
    for m in models:
        r1 = next((r["test_acc"] for r in rows
                   if r["model_name"] == m and r["target"] == "r1" and r["probe_type"] == "mlp"), 0.0)
        r2 = next((r["test_acc"] for r in rows
                   if r["model_name"] == m and r["target"] == "r2" and r["probe_type"] == "mlp"), 0.0)
        ret_acc.append(0.5 * (r1 + r2))
    ax.bar(x - width / 2, sum_acc, width, label="sum (MLP probe)", color="seagreen")
    ax.bar(x + width / 2, ret_acc, width, label="mean(r1,r2) (MLP probe)", color="purple")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(1 / 23, color="red", linestyle=":", linewidth=1, label="chance (1/23)")
    ax.set_title("Retrieval (r1, r2) vs sum probe accuracy")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(plot_dir, "retrieval_vs_sum_probe_accuracy.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"  wrote {p}")


def write_report(rows: list[dict], out_dir: str) -> None:
    lines: list[str] = []
    lines.append("# Hybrid Frozen Probe Diagnostics (p=23, m=3)")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("Frozen-transformer probes on the EQ-position residual stream.")
    lines.append(f"Linear probe = `Wz+b`; MLP probe = `W2 GELU(W1 z+b1)+b2` with hidden={PROBE_HIDDEN_DIM}.")
    lines.append("")
    lines.append("## 2. Per-model probe results (test accuracy)")
    models = sorted({r["model_name"] for r in rows})
    lines.append("| model | layers | linear sum | MLP sum | linear r1 | MLP r1 | linear r2 | MLP r2 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in models:
        layers = next((r["layers"] for r in rows if r["model_name"] == m), "?")
        def _acc(t, p):
            return next((r["test_acc"] for r in rows
                         if r["model_name"] == m and r["target"] == t and r["probe_type"] == p), None)
        cells = []
        for t in ("sum", "r1", "r2"):
            for p in ("linear", "mlp"):
                v = _acc(t, p)
                cells.append("n/a" if v is None else f"{v:.3f}")
        lines.append(f"| {m} | {layers} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 3. Interpretation")
    lines.append("Apply the diagnostic decision tree from the handoff doc:")
    lines.append("- If r1/r2 probes fail → final residual lacks queried values → retrieval/query-conditioning failure.")
    lines.append("- If r1/r2 probes succeed but sum probe fails → retrieval is present, sum is not computed.")
    lines.append("- If MLP sum probe succeeds but linear sum probe fails → information present but not linearly accessible.")
    lines.append("- If linear sum probe succeeds → readout is the bottleneck.")
    lines.append("")
    lines.append("## 4. Files")
    lines.append("- `probe_all_results.csv`")
    lines.append("- `probe_summary.csv`")
    lines.append("- `plots/probe_accuracy_by_target.png`")
    lines.append("- `plots/linear_vs_mlp_probe_accuracy.png`")
    lines.append("- `plots/retrieval_vs_sum_probe_accuracy.png`")
    p = os.path.join(out_dir, "hybrid_probe_diagnostics_report_for_chatgpt.md")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/hybrid_probe_diagnostics")
    parser.add_argument("--checkpoint_dirs", nargs="+", default=None,
                        help="Override candidate checkpoint dirs (relative to repo root).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--probe_epochs", type=int, default=DEFAULT_PROBE_EPOCHS)
    args = parser.parse_args()
    global _PROBE_EPOCHS
    _PROBE_EPOCHS = args.probe_epochs
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    cand = args.checkpoint_dirs or CANDIDATE_DIRS
    seen: set[str] = set()
    rows_all: list[dict] = []
    for rel in cand:
        d = os.path.join(PROJECT_ROOT, rel)
        if not os.path.isdir(d):
            print(f"  (skip — missing dir): {rel}")
            continue
        sig = model_dir_signature(d)
        if sig in seen:
            continue
        seen.add(sig)
        loaded = load_model_for_dir(d, device)
        if loaded is None:
            print(f"  (skip — no checkpoint): {rel}")
            continue
        model, meta, ckpt_path, ckpt_type = loaded
        run_label = os.path.basename(rel)
        print(f"\n=== probing {run_label} (n_layers={meta['n_layers']}, ckpt={ckpt_type}) ===")
        try:
            rows = diagnose_one(model, meta, device, run_label)
        except Exception:
            print(f"  ERROR while probing {run_label}:")
            traceback.print_exc()
            continue
        for r in rows:
            r["checkpoint_path"] = ckpt_path
            r["checkpoint_type"] = ckpt_type
        rows_all.extend(rows)

    if not rows_all:
        print("\nNo probe results — no checkpoints succeeded.")
        return

    write_results(rows_all, out_dir)
    make_plots(rows_all, out_dir)
    write_report(rows_all, out_dir)
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
