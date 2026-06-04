"""
analysis/erank_training_time_4hooks_modadd.py

Training-time eRank trajectories for the standard 3-layer transformer
trained on modular addition (run: normal_seed0, no freezing).

Produces four plots, each with eRank on the left y-axis and val/test
accuracy on the right y-axis vs. epoch:
  1. blocks.0.hook_attn_out   — layer 0 attention output (pre-residual-add)
  2. blocks.0.hook_mlp_out    — layer 0 MLP output (pre-residual-add)
  3. blocks.0.hook_resid_post — residual stream after MLP 0's contribution
  4. blocks.2.hook_resid_post — residual stream after MLP 2's contribution

Checkpoints sampled: every 100 epochs (`periodic_full_ep*.pt`) plus the
event checkpoints captured at grokking milestones. eRank is computed on
500 held-out val examples at the prediction position (pos=2), matching
the existing layer0-freeze pipeline.
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from analysis.erank import compute_erank
from data.splits import get_modular_addition_dataloaders_3way
from models.transformer import create_standard_transformer


RUN_DIR = os.path.join(
    PROJECT_ROOT, "outputs/layer0_freeze_modadd/normal_seed0"
)
CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")
OUT_DIR = os.path.join(RUN_DIR, "training_time_4hooks")

# Match the trainer's normal_seed0 settings (config_used.yaml)
P = 113
TRAIN_FRAC = 0.3
VAL_FRAC_OF_TRAIN = 0.2
BATCH_SIZE = 256
SEED = 0
N_LAYERS = 3
D_MODEL = 128
N_HEADS = 4
D_MLP = 512
ERANK_N_EXAMPLES = 500
PRED_POS = 2

# graph slug -> (hook name, plot title)
HOOKS: dict[str, tuple[str, str]] = {
    "graph1_layer0_attn_out": (
        "blocks.0.hook_attn_out",
        "Layer 0 attention output (pre-residual-add)",
    ),
    "graph2_layer0_mlp_out": (
        "blocks.0.hook_mlp_out",
        "Layer 0 MLP output (pre-residual-add)",
    ),
    "graph3_layer0_resid_post": (
        "blocks.0.hook_resid_post",
        "Residual stream after MLP 0 contribution",
    ),
    "graph4_layer2_resid_post": (
        "blocks.2.hook_resid_post",
        "Residual stream after MLP 2 contribution (final)",
    ),
    "graph5_layer1_attn_out": (
        "blocks.1.hook_attn_out",
        "Layer 1 attention output (pre-residual-add)",
    ),
    "graph6_layer1_mlp_out": (
        "blocks.1.hook_mlp_out",
        "Layer 1 MLP output (pre-residual-add)",
    ),
    "graph7_layer2_attn_out": (
        "blocks.2.hook_attn_out",
        "Layer 2 attention output (pre-residual-add)",
    ),
    "graph8_layer2_mlp_out": (
        "blocks.2.hook_mlp_out",
        "Layer 2 MLP output (pre-residual-add)",
    ),
}


def collect_checkpoints() -> list[tuple[int, str, str]]:
    """Return sorted list of (epoch, label, path) for the ckpts we plot over."""
    out: list[tuple[int, str, str]] = []

    rip = os.path.join(RUN_DIR, "random_init_checkpoint.pt")
    if os.path.exists(rip):
        out.append((0, "random_init", rip))

    for fn in os.listdir(CKPT_DIR):
        full = os.path.join(CKPT_DIR, fn)
        if fn.startswith("periodic_full_ep") and fn.endswith(".pt"):
            ep = int(fn[len("periodic_full_ep"):-3])
            out.append((ep, "periodic_full", full))
        elif fn.startswith("event_") and fn.endswith(".pt"):
            ep = int(fn[:-3].split("_ep")[-1])
            out.append((ep, fn[:-3], full))

    out.sort(key=lambda r: (r[0], r[1]))
    return out


def extract_eranks(model, val_loader, device) -> dict[str, float]:
    hook_names = [h for h, _ in HOOKS.values()]
    accum: dict[str, list[torch.Tensor]] = {n: [] for n in hook_names}
    name_set = set(hook_names)
    collected = 0
    with torch.no_grad():
        for inputs, _labels in val_loader:
            if collected >= ERANK_N_EXAMPLES:
                break
            inputs = inputs.to(device)
            B = inputs.size(0)
            remaining = ERANK_N_EXAMPLES - collected
            if B > remaining:
                inputs = inputs[:remaining]
                B = remaining
            _logits, cache = model.run_with_cache(
                inputs,
                names_filter=lambda n: n in name_set,
                return_type="logits",
            )
            for n in hook_names:
                accum[n].append(cache[n][:, PRED_POS, :].detach().cpu())
            collected += B

    eranks: dict[str, float] = {}
    for n in hook_names:
        A = torch.cat(accum[n], dim=0)[:ERANK_N_EXAMPLES]
        eranks[n] = compute_erank(A)
    return eranks


def read_training_log(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epochs, val_acc, test_acc = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                ep = int(row["epoch"])
                va = float(row["val_acc"])
                ta = float(row["test_acc"])
            except (KeyError, ValueError):
                continue
            epochs.append(ep)
            val_acc.append(va)
            test_acc.append(ta)
    return np.array(epochs), np.array(val_acc), np.array(test_acc)


def plot_one(
    slug: str,
    hook: str,
    title: str,
    erank_epochs: np.ndarray,
    erank_vals: np.ndarray,
    log_ep: np.ndarray,
    log_val_acc: np.ndarray,
    log_test_acc: np.ndarray,
    out_path: str,
) -> None:
    fig, ax_e = plt.subplots(figsize=(10.5, 5.8))

    color_erank = "#1f77b4"
    ax_e.plot(
        erank_epochs, erank_vals,
        "-o", ms=3.2, lw=1.4, color=color_erank, label="eRank",
    )
    ax_e.set_xlabel("Epoch")
    ax_e.set_ylabel("eRank", color=color_erank)
    ax_e.tick_params(axis="y", labelcolor=color_erank)
    ax_e.set_xlim(0, 5000)
    ax_e.grid(alpha=0.25)

    ax_l = ax_e.twinx()
    ax_l.plot(
        log_ep, log_val_acc,
        "-", lw=1.3, color="#ff7f0e", label="val accuracy", alpha=0.95,
    )
    ax_l.plot(
        log_ep, log_test_acc,
        "-", lw=1.3, color="#2ca02c", label="test accuracy", alpha=0.95,
    )
    ax_l.set_ylabel("Accuracy")
    ax_l.set_ylim(-0.02, 1.02)

    lines_1, labels_1 = ax_e.get_legend_handles_labels()
    lines_2, labels_2 = ax_l.get_legend_handles_labels()
    ax_e.legend(
        lines_1 + lines_2, labels_1 + labels_2,
        loc="upper right", framealpha=0.92,
    )

    plt.title(
        f"{title}\n"
        f"3-layer modular-addition transformer (normal_seed0)  —  hook: {hook}"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[erank-4hooks] device={device}")

    # Deterministic val_loader (same split as training run, seed=0)
    _train_loader, val_loader, _test_loader, _split_meta = (
        get_modular_addition_dataloaders_3way(
            p=P,
            train_frac=TRAIN_FRAC,
            val_frac_of_train=VAL_FRAC_OF_TRAIN,
            batch_size=BATCH_SIZE,
            seed=SEED,
        )
    )

    model = create_standard_transformer(
        task="modular_addition",
        cfg_overrides={
            "mod_p": P,
            "n_layers": N_LAYERS,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "d_mlp": D_MLP,
            "act_fn": "gelu",
        },
        seed=SEED,
    ).to(device)

    hook_names = [h for h, _ in HOOKS.values()]
    csv_path = os.path.join(OUT_DIR, "erank_4hooks_by_checkpoint.csv")

    # Skip recomputation if the eRank CSV already covers every requested hook.
    cache_ok = False
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            header = next(csv.reader(f))
        needed = {f"erank::{n}" for n in hook_names}
        cache_ok = needed.issubset(set(header))

    if cache_ok:
        rows: list[dict] = []
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                row = {"epoch": int(r["epoch"]), "label": r["label"]}
                for n in hook_names:
                    row[f"erank::{n}"] = float(r[f"erank::{n}"])
                rows.append(row)
        print(f"[erank-4hooks] reused cached eRanks from {csv_path} "
              f"({len(rows)} rows)")
        ckpts = []
    else:
        ckpts = collect_checkpoints()
        print(f"[erank-4hooks] {len(ckpts)} checkpoints to process")
        rows = []

    for i, (ep, label, path) in enumerate(ckpts):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd)
        model.eval()

        eranks = extract_eranks(model, val_loader, device)
        row = {"epoch": ep, "label": label}
        for n in hook_names:
            row[f"erank::{n}"] = eranks[n]
        rows.append(row)

        if (i + 1) % 25 == 0 or i + 1 == len(ckpts):
            short = " ".join(
                f"{n.split('.', 2)[-1]}={eranks[n]:.2f}" for n in hook_names
            )
            print(f"  [{i+1:>4}/{len(ckpts)}] ep={ep:<6} {label:<24} {short}")

    rows.sort(key=lambda r: r["epoch"])

    # Persist CSV (only if we recomputed)
    if ckpts:
        fieldnames = ["epoch", "label"] + [f"erank::{n}" for n in hook_names]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[erank-4hooks] wrote {csv_path}")

    # Accuracy series from training_log.csv
    log_ep, log_val_acc, log_test_acc = read_training_log(
        os.path.join(RUN_DIR, "training_log.csv")
    )

    erank_epochs = np.array([r["epoch"] for r in rows])
    for slug, (hook, title) in HOOKS.items():
        erank_vals = np.array([r[f"erank::{hook}"] for r in rows])
        out_png = os.path.join(OUT_DIR, f"{slug}.png")
        plot_one(
            slug, hook, title,
            erank_epochs, erank_vals,
            log_ep, log_val_acc, log_test_acc,
            out_png,
        )
        print(f"[erank-4hooks] wrote {out_png}")


if __name__ == "__main__":
    main()
