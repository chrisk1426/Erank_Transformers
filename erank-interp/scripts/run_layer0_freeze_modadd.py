"""
scripts/run_layer0_freeze_modadd.py

Per-run trainer for the layer-0 freezing experiment on modular addition
(see claude_code_layer0_freeze_modadd_experiment_3seeds_7gpu.md).

Trains a 3-layer HookedTransformer on c = (a + b) mod 113 with optional
freezing of layer-0 attention and/or MLP. Frozen components are kept in the
forward pass but their parameters are set requires_grad=False and excluded
from the optimizer parameter group.

Outputs (under <output_root>/<run_name>/):
    config_used.yaml
    trainability_summary.txt
    training_log.csv
    checkpoint_manifest.csv
    checkpoints/                       periodic + event ckpts
    random_init_checkpoint.pt
    best_val_checkpoint.pt
    final_checkpoint.pt
    erank_by_checkpoint.csv            per-layer eRank at every saved checkpoint
    erank_summary_by_checkpoint.csv    aggregate per-checkpoint row
    same_sum_clustering.csv            optional
    training.png
    run_report.md
    run_summary.json
    status.txt
    logs/train.log                     captured by the launcher
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
import yaml
from tqdm import tqdm

from analysis.erank import compute_erank
from data.splits import get_modular_addition_dataloaders_3way
from models.transformer import create_standard_transformer

# ---------------------------------------------------------------------------
# Defaults (mirror the handoff doc)
# ---------------------------------------------------------------------------

P = 113
N_LAYERS = 3
D_MODEL = 128
N_HEADS = 4
D_HEAD = 32
D_MLP = 512
LR = 3e-4
WEIGHT_DECAY = 1.0
BATCH_SIZE = 256
MAX_EPOCHS = 20000
TRAIN_FRAC = 0.30
VAL_FRAC_OF_TRAIN = 0.20

# Checkpoint cadence: model-only every 10 epochs (heavy), full every 100,
# plus event checkpoints. The launcher can override via CLI to coarser.
DEFAULT_CKPT_EVERY = 10
DEFAULT_FULL_CKPT_EVERY = 100

EVAL_EVERY = 25
EVENT_VAL_THRESHOLDS = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
TRAIN_EVENT_THRESHOLD = 0.99
ERANK_N_EXAMPLES = 500
GROK_THRESH_LOW = 0.90
GROK_THRESH_HIGH = 0.95
SUSTAINED_EVALS_EARLY_STOP = 200  # at val>=0.95, stop after this many evals


def _resolve_freeze_prefixes(freeze_a0: bool, freeze_m0: bool) -> list[str]:
    prefixes: list[str] = []
    if freeze_a0:
        prefixes.append("blocks.0.attn")
    if freeze_m0:
        prefixes.append("blocks.0.mlp")
    return prefixes


def apply_freeze_mask(model: nn.Module, freeze_prefixes: list[str]) -> dict:
    """
    Set requires_grad=False on every parameter whose name starts with any
    prefix in freeze_prefixes. Returns a per-component summary dict.
    """
    summary: dict = {}
    for name, p in model.named_parameters():
        frozen = any(name.startswith(pre) for pre in freeze_prefixes)
        if frozen:
            p.requires_grad = False
        component = name.rsplit(".", 1)[0]
        s = summary.setdefault(component, {"params": 0, "trainable_params": 0,
                                            "frozen": False})
        s["params"] += p.numel()
        if not frozen:
            s["trainable_params"] += p.numel()
        s["frozen"] = s["frozen"] or frozen
    return summary


def write_trainability_summary(summary: dict, path: str,
                               variant: str, seed: int) -> None:
    """Human-readable + machine-parseable trainability table."""
    rows = []
    total_params = 0
    total_trainable = 0
    for component, s in sorted(summary.items()):
        rows.append({
            "component": component,
            "num_params": s["params"],
            "num_trainable_params": s["trainable_params"],
            "frozen_or_trainable": "frozen" if s["frozen"] else "trainable",
        })
        total_params += s["params"]
        total_trainable += s["trainable_params"]
    with open(path, "w") as f:
        f.write(f"variant: {variant}\nseed: {seed}\n")
        f.write(f"total_params: {total_params}\n")
        f.write(f"total_trainable_params: {total_trainable}\n")
        f.write(f"total_frozen_params: {total_params - total_trainable}\n\n")
        f.write(f"{'component':<28} {'num_params':>12} {'trainable':>12}  frozen\n")
        for r in rows:
            f.write(f"{r['component']:<28} {r['num_params']:>12} "
                    f"{r['num_trainable_params']:>12}  "
                    f"{r['frozen_or_trainable']}\n")


def evaluate(model, loader, device, criterion) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)[:, 2, :]
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_correct += (logits.argmax(dim=-1) == labels).sum().item()
            n += labels.size(0)
    return total_loss / max(len(loader), 1), n_correct / n


def train_one_epoch(model, loader, optimizer, device, criterion) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    n_correct = 0
    n = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)[:, 2, :]
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_correct += (logits.argmax(dim=-1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / max(len(loader), 1), n_correct / n


def save_checkpoint(path: str, model, optimizer, epoch: int,
                    *, model_only: bool = False, extra: dict | None = None) -> None:
    state = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "is_model_only": model_only,
    }
    if not model_only:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)


@torch.no_grad()
def extract_residual_activations_hooked(
    model, loader, device, n_examples: int, pred_position: int,
) -> dict[str, torch.Tensor]:
    """
    Run examples through a HookedTransformer and collect resid_pre / resid_mid /
    resid_post at the prediction position for every layer.
    """
    model.eval()
    n_layers = model.cfg.n_layers
    names = [f"blocks.{i}.hook_resid_{tag}" for i in range(n_layers)
             for tag in ("pre", "mid", "post")]
    accum: dict[str, list[torch.Tensor]] = {n: [] for n in names}
    collected = 0
    for inputs, _ in loader:
        if collected >= n_examples:
            break
        inputs = inputs.to(device)
        B = inputs.size(0)
        remaining = n_examples - collected
        if B > remaining:
            inputs = inputs[:remaining]
            B = remaining
        _, cache = model.run_with_cache(inputs, names_filter=names)
        for n in names:
            accum[n].append(cache[n][:, pred_position, :].detach().cpu())
        collected += B
    return {n: torch.cat(parts, dim=0)[:n_examples] for n, parts in accum.items()}


def erank_profile(activations: dict[str, torch.Tensor], n_layers: int) -> dict:
    """Per-layer eRank pre/mid/post and delta_attn / delta_mlp."""
    out = {"erank": {}, "delta_attn": {}, "delta_mlp": {}}
    for i in range(n_layers):
        pre = compute_erank(activations[f"blocks.{i}.hook_resid_pre"])
        mid = compute_erank(activations[f"blocks.{i}.hook_resid_mid"])
        post = compute_erank(activations[f"blocks.{i}.hook_resid_post"])
        out["erank"][i] = {"pre": pre, "mid": mid, "post": post}
        out["delta_attn"][i] = mid - pre
        out["delta_mlp"][i] = post - mid
    return out


def same_sum_clustering(resid: torch.Tensor, labels: torch.Tensor,
                        ) -> tuple[float, float, float, int]:
    A = resid.detach().float().cpu().numpy()
    y = labels.detach().cpu().numpy()
    classes = np.unique(y)
    centroids: list[np.ndarray] = []
    within_sq_sum = 0.0
    n_total = 0
    for c in classes:
        mask = (y == c)
        rows = A[mask]
        if rows.shape[0] == 0:
            continue
        centroid = rows.mean(axis=0)
        centroids.append(centroid)
        within_sq_sum += float(((rows - centroid) ** 2).sum())
        n_total += rows.shape[0]
    if n_total == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    D_within = within_sq_sum / n_total
    cent = np.stack(centroids, axis=0)
    C = cent.shape[0]
    if C < 2:
        return (D_within, float("nan"), float("nan"), n_total)
    diffs = cent[:, None, :] - cent[None, :, :]
    sq = (diffs * diffs).sum(axis=-1)
    iu = np.triu_indices(C, k=1)
    D_between = float(sq[iu].sum() * 2 / (C * (C - 1)))
    ratio = D_within / D_between if D_between > 0 else float("nan")
    return (D_within, D_between, ratio, n_total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=["normal", "freeze_A0", "freeze_M0", "freeze_A0_M0"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--freeze-a0", action="store_true")
    parser.add_argument("--freeze-m0", action="store_true")
    parser.add_argument("--output-root", default="outputs/layer0_freeze_modadd")
    parser.add_argument("--run-name", default=None,
                        help="Defaults to '{variant}_seed{seed}'")
    parser.add_argument("--base-init-dir", default=None,
                        help="Directory holding base_init_seed{N}.pt; if a file exists, "
                             "the model loads from it for shared-init within a seed.")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CKPT_EVERY)
    parser.add_argument("--full-checkpoint-every", type=int,
                        default=DEFAULT_FULL_CKPT_EVERY)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--p", type=int, default=P)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val-frac-of-train", type=float, default=VAL_FRAC_OF_TRAIN)
    parser.add_argument("--sustained-early-stop-evals", type=int,
                        default=SUSTAINED_EVALS_EARLY_STOP)
    parser.add_argument("--same-sum-clustering", action="store_true", default=True)
    parser.add_argument("--no-same-sum-clustering", action="store_false",
                        dest="same_sum_clustering")
    args = parser.parse_args()

    run_name = args.run_name or f"{args.variant}_seed{args.seed}"
    run_dir = os.path.join(PROJECT_ROOT, args.output_root, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(run_dir, "status.txt"), "w") as f:
        f.write("RUNNING\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{run_name}] device={device}  variant={args.variant}  seed={args.seed}  "
          f"freeze_a0={args.freeze_a0}  freeze_m0={args.freeze_m0}")

    # --- Data ---
    train_loader, val_loader, test_loader, split_meta = get_modular_addition_dataloaders_3way(
        p=args.p,
        train_frac=args.train_frac,
        val_frac_of_train=args.val_frac_of_train,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    with open(os.path.join(run_dir, "split_metadata.json"), "w") as f:
        json.dump(split_meta, f, indent=2)

    # --- Model ---
    model = create_standard_transformer(
        task="modular_addition",
        cfg_overrides={
            "mod_p": args.p,
            "n_layers": N_LAYERS,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "d_mlp": D_MLP,
            "act_fn": "gelu",
        },
        seed=args.seed,
    ).to(device)

    # Shared-init: load base_init_seed{N}.pt if available; otherwise create + save it.
    base_init_dir = args.base_init_dir or os.path.join(
        PROJECT_ROOT, args.output_root)
    os.makedirs(base_init_dir, exist_ok=True)
    base_init_path = os.path.join(base_init_dir, f"base_init_seed{args.seed}.pt")
    if os.path.exists(base_init_path):
        ckpt = torch.load(base_init_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd)
        print(f"[{run_name}] loaded shared base init from {base_init_path}")
    else:
        torch.save({"model_state_dict": model.state_dict(), "epoch": 0,
                    "note": "shared base init for this seed"}, base_init_path)
        print(f"[{run_name}] wrote shared base init to {base_init_path}")

    # Apply freeze masks.
    freeze_prefixes = _resolve_freeze_prefixes(args.freeze_a0, args.freeze_m0)
    summary = apply_freeze_mask(model, freeze_prefixes)
    write_trainability_summary(
        summary,
        os.path.join(run_dir, "trainability_summary.txt"),
        variant=args.variant, seed=args.seed,
    )

    # Optimizer over trainable params only.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[{run_name}] params total={n_total:,} trainable={n_trainable:,} "
          f"frozen={n_total - n_trainable:,}")

    # Save random-init.
    random_init_path = os.path.join(run_dir, "random_init_checkpoint.pt")
    save_checkpoint(random_init_path, model, optimizer, epoch=0)

    config_used = {
        "task": "modular_addition",
        "modulus": args.p,
        "seed": args.seed,
        "variant": args.variant,
        "freeze_A0": args.freeze_a0,
        "freeze_M0": args.freeze_m0,
        "n_layers": N_LAYERS,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "d_head": D_HEAD,
        "d_mlp": D_MLP,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "eval_every": args.eval_every,
        "checkpoint_every": args.checkpoint_every,
        "full_checkpoint_every": args.full_checkpoint_every,
        "train_frac": args.train_frac,
        "val_frac_of_train": args.val_frac_of_train,
        "n_total": split_meta["n_total"],
        "n_train": split_meta["n_train"],
        "n_val": split_meta["n_val"],
        "n_test": split_meta["n_test"],
        "n_params": n_total,
        "n_trainable_params": n_trainable,
        "n_frozen_params": n_total - n_trainable,
        "run_name": run_name,
        "sustained_early_stop_evals": args.sustained_early_stop_evals,
    }
    with open(os.path.join(run_dir, "config_used.yaml"), "w") as f:
        yaml.dump(config_used, f, default_flow_style=False)

    # --- Training loop ---
    history: dict[str, list] = {
        "epoch": [], "step": [],
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [], "test_acc": [], "test_loss": [],
        "best_val_acc_so_far": [], "elapsed_time_seconds": [],
    }
    checkpoint_manifest: list[dict] = []
    erank_rows: list[dict] = []          # per-checkpoint × per-layer
    erank_summary_rows: list[dict] = []  # per-checkpoint aggregate
    same_sum_rows: list[dict] = []

    best_val_acc = -1.0
    best_val_epoch: int | None = None
    best_val_path = os.path.join(run_dir, "best_val_checkpoint.pt")
    final_path = os.path.join(run_dir, "final_checkpoint.pt")
    event_thresholds = list(EVENT_VAL_THRESHOLDS)
    train_event_done = False
    sustained_high = 0
    tau_0_90: int | None = None
    tau_0_95: int | None = None
    SAME_SUM_KEY_LABELS = {
        "random_init", "first_train_acc_ge_0.99",
        "first_val_acc_ge_0.90", "first_val_acc_ge_0.95",
        "best_val", "final",
    }

    # --- Helper to compute eRank + clustering at a saved checkpoint label ---
    def _compute_metrics_at(label: str, ckpt_path: str, epoch: int,
                            train_acc=None, val_acc=None,
                            test_acc_at_best_val=None) -> None:
        """Compute eRank profile (+ optionally same-sum) for a checkpoint."""
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            eval_model = create_standard_transformer(
                task="modular_addition",
                cfg_overrides={
                    "mod_p": args.p, "n_layers": N_LAYERS, "d_model": D_MODEL,
                    "n_heads": N_HEADS, "d_mlp": D_MLP, "act_fn": "gelu",
                },
                seed=args.seed,
            ).to(device)
            eval_model.load_state_dict(sd)
            acts = extract_residual_activations_hooked(
                eval_model, val_loader, device, n_examples=ERANK_N_EXAMPLES,
                pred_position=2,
            )
            prof = erank_profile(acts, n_layers=N_LAYERS)
            for li in range(N_LAYERS):
                erank_rows.append({
                    "run_name": run_name, "variant": args.variant,
                    "seed": args.seed,
                    "freeze_A0": int(args.freeze_a0), "freeze_M0": int(args.freeze_m0),
                    "epoch": epoch, "checkpoint_label": label,
                    "train_acc": "" if train_acc is None else train_acc,
                    "val_acc": "" if val_acc is None else val_acc,
                    "test_acc_at_best_val_if_applicable":
                        "" if test_acc_at_best_val is None else test_acc_at_best_val,
                    "layer": li,
                    "erank_pre": prof["erank"][li]["pre"],
                    "erank_mid": prof["erank"][li]["mid"],
                    "erank_post": prof["erank"][li]["post"],
                    "delta_attn": prof["delta_attn"][li],
                    "delta_mlp": prof["delta_mlp"][li],
                })
            final_rep_erank = prof["erank"][N_LAYERS - 1]["post"]
            sum_da = float(sum(prof["delta_attn"].values()))
            sum_dm = float(sum(prof["delta_mlp"].values()))
            erank_summary_rows.append({
                "run_name": run_name, "variant": args.variant, "seed": args.seed,
                "epoch": epoch, "checkpoint_label": label,
                "train_acc": "" if train_acc is None else train_acc,
                "val_acc": "" if val_acc is None else val_acc,
                "sum_delta_attn": sum_da,
                "sum_delta_mlp": sum_dm,
                "final_rep_erank": final_rep_erank,
                "layer0_delta_mlp": prof["delta_mlp"][0],
                "layer1_delta_mlp": prof["delta_mlp"][1],
                "layer2_delta_mlp": prof["delta_mlp"][2],
                "layer0_delta_attn": prof["delta_attn"][0],
                "layer1_delta_attn": prof["delta_attn"][1],
                "layer2_delta_attn": prof["delta_attn"][2],
            })
            # Same-sum clustering at key labels.
            if args.same_sum_clustering and label in SAME_SUM_KEY_LABELS:
                labels_list: list[torch.Tensor] = []
                got = 0
                for _x, y in val_loader:
                    labels_list.append(y)
                    got += y.size(0)
                    if got >= ERANK_N_EXAMPLES:
                        break
                labels_t = torch.cat(labels_list, dim=0)[:ERANK_N_EXAMPLES]
                final_resid = acts[f"blocks.{N_LAYERS-1}.hook_resid_post"][:labels_t.shape[0]]
                D_w, D_b, ratio, n_ex = same_sum_clustering(final_resid, labels_t)
                same_sum_rows.append({
                    "run_name": run_name, "variant": args.variant, "seed": args.seed,
                    "epoch": epoch, "checkpoint_label": label,
                    "train_acc": "" if train_acc is None else train_acc,
                    "val_acc": "" if val_acc is None else val_acc,
                    "D_within": D_w, "D_between": D_b,
                    "within_between_ratio": ratio, "num_examples": n_ex,
                    "analysis_set": "val",
                })
        except Exception:
            print(f"  [metrics @ {label} ep {epoch}] WARNING")
            traceback.print_exc()

    def _record_ckpt(label: str, epoch: int, path: str, is_model_only: bool,
                     train_acc=None, val_acc=None, train_loss=None, val_loss=None):
        checkpoint_manifest.append({
            "run_name": run_name, "variant": args.variant, "seed": args.seed,
            "epoch": int(epoch), "checkpoint_label": label,
            "train_acc": "" if train_acc is None else train_acc,
            "val_acc": "" if val_acc is None else val_acc,
            "train_loss": "" if train_loss is None else train_loss,
            "val_loss": "" if val_loss is None else val_loss,
            "checkpoint_path": path,
            "is_model_only": int(is_model_only),
            "is_full_checkpoint": int(not is_model_only),
        })

    _record_ckpt("random_init", 0, random_init_path, is_model_only=False)
    _compute_metrics_at("random_init", random_init_path, 0)

    t_start = time.time()
    step = 0
    last_epoch = 0
    last_train_acc = float("nan")
    last_train_loss = float("nan")
    last_val_acc = float("nan")
    last_val_loss = float("nan")
    for epoch in tqdm(range(1, args.max_epochs + 1), desc=f"[{run_name}]"):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, criterion)
        step += max(1, len(train_loader))
        last_epoch = epoch
        last_train_acc = train_acc
        last_train_loss = train_loss

        do_eval = (epoch % args.eval_every == 0) or (epoch == 1) or (epoch == args.max_epochs)
        if do_eval:
            val_loss, val_acc = evaluate(model, val_loader, device, criterion)
            test_loss, test_acc = evaluate(model, test_loader, device, criterion)
            last_val_acc = val_acc
            last_val_loss = val_loss
            history["epoch"].append(epoch)
            history["step"].append(step)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["test_acc"].append(test_acc)
            history["best_val_acc_so_far"].append(max(best_val_acc, val_acc))
            history["elapsed_time_seconds"].append(time.time() - t_start)

            if epoch % 500 == 0 or epoch == 1:
                print(f"  ep {epoch:6d} | tr_loss {train_loss:.4f} val_loss {val_loss:.4f} "
                      f"| tr_acc {train_acc:.3f} val_acc {val_acc:.3f} test_acc {test_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch
                save_checkpoint(best_val_path, model, optimizer, epoch,
                                extra={"val_acc": val_acc, "val_loss": val_loss})

            if not train_event_done and train_acc >= TRAIN_EVENT_THRESHOLD:
                p_evt = os.path.join(ckpt_dir, f"event_first_train_acc_ge_0.99_ep{epoch}.pt")
                save_checkpoint(p_evt, model, optimizer, epoch,
                                extra={"event": f"train_acc>={TRAIN_EVENT_THRESHOLD}"})
                _record_ckpt("first_train_acc_ge_0.99", epoch, p_evt,
                             is_model_only=False,
                             train_acc=train_acc, val_acc=val_acc,
                             train_loss=train_loss, val_loss=val_loss)
                _compute_metrics_at("first_train_acc_ge_0.99", p_evt, epoch,
                                    train_acc=train_acc, val_acc=val_acc)
                train_event_done = True

            for thr in list(event_thresholds):
                if val_acc >= thr:
                    label = f"first_val_acc_ge_{thr:.2f}"
                    p_evt = os.path.join(ckpt_dir, f"event_{label}_ep{epoch}.pt")
                    save_checkpoint(p_evt, model, optimizer, epoch,
                                    extra={"event": f"val_acc>={thr}"})
                    _record_ckpt(label, epoch, p_evt, is_model_only=False,
                                 train_acc=train_acc, val_acc=val_acc,
                                 train_loss=train_loss, val_loss=val_loss)
                    _compute_metrics_at(label, p_evt, epoch,
                                        train_acc=train_acc, val_acc=val_acc)
                    event_thresholds.remove(thr)

            if tau_0_90 is None and val_acc >= GROK_THRESH_LOW:
                tau_0_90 = epoch
            if tau_0_95 is None and val_acc >= GROK_THRESH_HIGH:
                tau_0_95 = epoch
            sustained_high = sustained_high + 1 if val_acc >= GROK_THRESH_HIGH else 0

        # Model-only periodic ckpt (light).
        if epoch % args.checkpoint_every == 0 and epoch % args.full_checkpoint_every != 0:
            p_per = os.path.join(ckpt_dir, f"periodic_modelonly_ep{epoch}.pt")
            save_checkpoint(p_per, model, optimizer, epoch, model_only=True,
                            extra={"event": "periodic_modelonly"})
            _record_ckpt("periodic_modelonly", epoch, p_per, is_model_only=True,
                         train_acc=last_train_acc, val_acc=last_val_acc,
                         train_loss=last_train_loss, val_loss=last_val_loss)
            # Don't compute eRank for every model-only periodic (too slow);
            # compute on full periodic only.

        if epoch % args.full_checkpoint_every == 0:
            p_per = os.path.join(ckpt_dir, f"periodic_full_ep{epoch}.pt")
            save_checkpoint(p_per, model, optimizer, epoch, model_only=False,
                            extra={"event": "periodic_full"})
            _record_ckpt("periodic_full", epoch, p_per, is_model_only=False,
                         train_acc=last_train_acc, val_acc=last_val_acc,
                         train_loss=last_train_loss, val_loss=last_val_loss)
            _compute_metrics_at("periodic_full", p_per, epoch,
                                train_acc=last_train_acc, val_acc=last_val_acc)

        if sustained_high >= args.sustained_early_stop_evals:
            print(f"\n[{run_name}] Early stop @ ep {epoch} — val_acc >= "
                  f"{GROK_THRESH_HIGH} for {args.sustained_early_stop_evals} evals.")
            break

        # Hard-cap stop for variants that grok and then oscillate around 0.95.
        # If we've ever hit the grokking threshold (tau_0_95 set), don't burn
        # GPU past 2*tau_0_95 + 1000 epochs waiting for a sustained streak.
        if tau_0_95 is not None and epoch >= 2 * tau_0_95 + 1000:
            print(f"\n[{run_name}] Hard-cap stop @ ep {epoch} — first grok at "
                  f"tau_0.95={tau_0_95}, hard cap = 2*tau + 1000 = "
                  f"{2 * tau_0_95 + 1000}.")
            break

    elapsed = time.time() - t_start

    save_checkpoint(final_path, model, optimizer, last_epoch)
    _record_ckpt("final", last_epoch, final_path, is_model_only=False,
                 train_acc=last_train_acc, val_acc=last_val_acc,
                 train_loss=last_train_loss, val_loss=last_val_loss)
    _compute_metrics_at("final", final_path, last_epoch,
                        train_acc=last_train_acc, val_acc=last_val_acc)
    if best_val_epoch is not None:
        # Read best val test_acc from history.
        best_test_acc = None
        best_test_loss = None
        if best_val_epoch in history["epoch"]:
            idx = history["epoch"].index(best_val_epoch)
            best_test_acc = history["test_acc"][idx]
            best_test_loss = history["test_loss"][idx]
        _record_ckpt("best_val", best_val_epoch, best_val_path, is_model_only=False,
                     val_acc=best_val_acc)
        _compute_metrics_at("best_val", best_val_path, best_val_epoch,
                            train_acc=None, val_acc=best_val_acc,
                            test_acc_at_best_val=best_test_acc)
    else:
        best_test_acc = None
        best_test_loss = None

    # --- Status label ---
    final_train_acc = history["train_acc"][-1] if history["train_acc"] else float("nan")
    final_val_acc = history["val_acc"][-1] if history["val_acc"] else float("nan")
    if best_val_acc >= 0.95:
        status = "GROKKED"
    elif final_train_acc >= 0.99 and best_val_acc < 0.20:
        status = "MEMORIZED_ONLY"
    elif final_train_acc < 0.90 and best_val_acc < 0.20:
        status = "FAILED_TO_FIT"
    else:
        status = "PARTIAL"

    # --- Write CSVs ---
    manifest_fields = [
        "run_name", "variant", "seed", "epoch", "checkpoint_label",
        "train_acc", "val_acc", "train_loss", "val_loss",
        "checkpoint_path", "is_model_only", "is_full_checkpoint",
    ]
    with open(os.path.join(run_dir, "checkpoint_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest_fields)
        w.writeheader()
        for r in checkpoint_manifest:
            w.writerow(r)

    training_log_fields = [
        "run_name", "variant", "seed", "epoch", "step",
        "train_loss", "val_loss", "test_loss",
        "train_acc", "val_acc", "test_acc",
        "best_val_acc_so_far", "learning_rate", "weight_decay",
        "elapsed_time_seconds",
    ]
    with open(os.path.join(run_dir, "training_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=training_log_fields)
        w.writeheader()
        for i, ep in enumerate(history["epoch"]):
            w.writerow({
                "run_name": run_name, "variant": args.variant, "seed": args.seed,
                "epoch": ep, "step": history["step"][i],
                "train_loss": history["train_loss"][i],
                "val_loss": history["val_loss"][i],
                "test_loss": history["test_loss"][i],
                "train_acc": history["train_acc"][i],
                "val_acc": history["val_acc"][i],
                "test_acc": history["test_acc"][i],
                "best_val_acc_so_far": history["best_val_acc_so_far"][i],
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "elapsed_time_seconds": history["elapsed_time_seconds"][i],
            })

    erank_fields = [
        "run_name", "variant", "seed", "freeze_A0", "freeze_M0",
        "epoch", "checkpoint_label", "train_acc", "val_acc",
        "test_acc_at_best_val_if_applicable",
        "layer", "erank_pre", "erank_mid", "erank_post",
        "delta_attn", "delta_mlp",
    ]
    with open(os.path.join(run_dir, "erank_by_checkpoint.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=erank_fields)
        w.writeheader()
        for r in erank_rows:
            w.writerow(r)

    summary_fields = [
        "run_name", "variant", "seed", "epoch", "checkpoint_label",
        "train_acc", "val_acc",
        "sum_delta_attn", "sum_delta_mlp", "final_rep_erank",
        "layer0_delta_mlp", "layer1_delta_mlp", "layer2_delta_mlp",
        "layer0_delta_attn", "layer1_delta_attn", "layer2_delta_attn",
    ]
    with open(os.path.join(run_dir, "erank_summary_by_checkpoint.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in erank_summary_rows:
            w.writerow(r)

    if same_sum_rows:
        ss_fields = [
            "run_name", "variant", "seed", "epoch", "checkpoint_label",
            "train_acc", "val_acc",
            "D_within", "D_between", "within_between_ratio",
            "num_examples", "analysis_set",
        ]
        with open(os.path.join(run_dir, "same_sum_clustering.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ss_fields)
            w.writeheader()
            for r in same_sum_rows:
                w.writerow(r)

    # --- Training plot ---
    try:
        ep_arr = history["epoch"]
        fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(11, 4))
        ax_l.plot(ep_arr, history["train_loss"], label="train")
        ax_l.plot(ep_arr, history["val_loss"], label="val")
        ax_l.set_xlabel("epoch"); ax_l.set_ylabel("loss"); ax_l.set_yscale("log")
        ax_l.set_title(f"{run_name} loss"); ax_l.legend()
        ax_a.plot(ep_arr, history["train_acc"], label="train")
        ax_a.plot(ep_arr, history["val_acc"], label="val")
        ax_a.plot(ep_arr, history["test_acc"], label="test", linestyle="--", alpha=0.6)
        ax_a.axhline(1/args.p, color="red", linestyle=":", linewidth=1,
                     label=f"chance (1/{args.p})")
        ax_a.axhline(0.90, color="purple", linestyle="--", linewidth=0.7, label="0.90")
        ax_a.axhline(0.95, color="purple", linestyle="-", linewidth=0.7, label="0.95")
        ax_a.set_xlabel("epoch"); ax_a.set_ylabel("accuracy"); ax_a.set_ylim(0, 1.05)
        ax_a.set_title(f"{run_name} accuracy"); ax_a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "training.png"), dpi=140)
        plt.close(fig)
    except Exception:
        traceback.print_exc()

    # --- Run report ---
    rep_lines = []
    rep_lines.append(f"# Run: {run_name}")
    rep_lines.append("")
    rep_lines.append(f"- variant: `{args.variant}`")
    rep_lines.append(f"- freeze A0: `{args.freeze_a0}`, freeze M0: `{args.freeze_m0}`")
    rep_lines.append(f"- seed: {args.seed}")
    rep_lines.append(f"- 3-layer standard transformer, p={args.p}")
    rep_lines.append(f"- params total/trainable/frozen = "
                     f"{n_total:,} / {n_trainable:,} / {n_total - n_trainable:,}")
    rep_lines.append(f"- ran {last_epoch} epochs in {elapsed/3600:.2f} h (cap {args.max_epochs})")
    rep_lines.append("")
    rep_lines.append(f"- best_val_acc = {best_val_acc:.4f} @ ep {best_val_epoch}")
    rep_lines.append(f"- test_acc@best_val = "
                     f"{best_test_acc if best_test_acc is None else f'{best_test_acc:.4f}'}")
    rep_lines.append(f"- tau_0.90: {tau_0_90 if tau_0_90 is not None else 'NA'}")
    rep_lines.append(f"- tau_0.95: {tau_0_95 if tau_0_95 is not None else 'NA'}")
    rep_lines.append(f"- status: **{status}**")
    rep_lines.append("")
    rep_lines.append("![training](training.png)")
    with open(os.path.join(run_dir, "run_report.md"), "w") as f:
        f.write("\n".join(rep_lines))

    # --- Run summary JSON ---
    summary_json = {
        "run_name": run_name,
        "variant": args.variant,
        "seed": args.seed,
        "freeze_A0": args.freeze_a0,
        "freeze_M0": args.freeze_m0,
        "best_val_acc": best_val_acc,
        "best_val_epoch": best_val_epoch,
        "test_acc_at_best_val": best_test_acc,
        "test_loss_at_best_val": best_test_loss,
        "final_train_acc": float(final_train_acc),
        "final_val_acc": float(final_val_acc),
        "final_epoch": int(last_epoch),
        "tau_0.90": tau_0_90,
        "tau_0.95": tau_0_95,
        "status": status,
        "n_params_total": int(n_total),
        "n_trainable_params": int(n_trainable),
        "n_frozen_params": int(n_total - n_trainable),
        "elapsed_seconds": float(elapsed),
    }
    with open(os.path.join(run_dir, "run_summary.json"), "w") as f:
        json.dump(summary_json, f, indent=2, default=float)

    with open(os.path.join(run_dir, "status.txt"), "w") as f:
        f.write(f"{status}\n")
    print(f"[{run_name}] DONE — {status} — outputs in {run_dir}")


if __name__ == "__main__":
    main()
