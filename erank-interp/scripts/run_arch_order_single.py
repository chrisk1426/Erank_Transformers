"""
scripts/run_arch_order_single.py

Per-run trainer for the balanced A/M architecture-order sweep
(claude_code_arch_order_sweep_6gpu_with_results_report.md).

Trains a BlockScheduleTransformer on modular addition with validation-based
checkpoint selection and emits the full set of artifacts required by the
handoff doc, including componentwise (per-block) eRank with the doc-specified
schema, spectrum metrics with top-k mass for k in {1,5,10,20}, and same-sum
clustering with explicit D_within / D_between columns.

Outputs (under <output_root>/runs/<run_id>/):
    config_used.yaml
    training_log.csv
    checkpoints/                   # periodic (every 500 ep) + event
    checkpoint_manifest.csv
    random_init_checkpoint.pt
    best_val_checkpoint.pt
    final_checkpoint.pt
    componentwise_erank.csv        # per-block deltas at multiple ckpts
    training_time_erank.csv        # per-epoch eRank summaries
    spectrum_metrics.csv           # top-k mass at best-val
    same_sum_clustering.csv        # D_within/D_between at random_init/best/final
    status.txt
    run_report.md
    logs/train.log                 # captured by the launcher
"""

from __future__ import annotations

import argparse
import csv
import json
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
from models.block_schedule_transformer import BlockScheduleTransformer

# ---------------------------------------------------------------------------
# Defaults (mirror the handoff doc)
# ---------------------------------------------------------------------------

P = 113
SEED = 0
D_MODEL = 128
N_HEADS = 4
D_HEAD = 32
D_MLP = 512
LR = 3e-4
WEIGHT_DECAY = 1.0
BATCH_SIZE = 256
MAX_EPOCHS = 50000
TRAIN_FRAC = 0.30
VAL_FRAC_OF_TRAIN = 0.20
PERIODIC_EVERY = 500
PERIODIC_ERANK_EVERY = 1000
EVAL_EVERY = 25
EVENT_VAL_THRESHOLDS = [0.05, 0.25, 0.50, 0.75, 0.95]
TRAIN_EVENT_THRESHOLD = 0.99
ERANK_N_EXAMPLES = 500
GROKKING_VAL_THRESHOLD = 0.95
SUSTAINED_EARLY_STOP_EVALS = 200  # consecutive evals at val >= 0.95 to early-stop
SPECTRUM_KS = [1, 5, 10, 20]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_schedule(s: str) -> list[str]:
    """
    Parse a schedule string into a list of A/M/AM tokens.

    For the balanced sweep, all schedules are pure A/M sequences (e.g. 'AAMM'),
    so by default each character is its own block. Combined 'AM' blocks must be
    expressed with explicit separators ('AM,AM,AM' or 'AM-AM-AM').
    """
    s = s.strip()
    if "," in s:
        return [tok.strip() for tok in s.split(",")]
    if "-" in s:
        return [tok.strip() for tok in s.split("-")]
    out: list[str] = []
    for c in s:
        if c in ("A", "M"):
            out.append(c)
        else:
            raise ValueError(f"Bad schedule char {c!r} in {s!r}")
    return out


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


def save_checkpoint(path: str, model, optimizer, epoch: int, extra: dict | None = None) -> None:
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "block_schedule": list(model.cfg.block_schedule),
        "model_kind": "BlockScheduleTransformer",
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def topk_singular_mass(activations: torch.Tensor, ks: list[int]) -> tuple[dict[int, float], int]:
    """Return ({k: top-k singular mass}, rank_nonzero)."""
    A = activations.detach().float().cpu().numpy()
    A = A - A.mean(axis=0)
    sigma = np.linalg.svd(A, full_matrices=False, compute_uv=False)
    total = float(sigma.sum())
    rank_nonzero = int((sigma > 1e-8).sum())
    if total == 0:
        return ({k: 0.0 for k in ks}, rank_nonzero)
    return ({k: float(sigma[:k].sum() / total) for k in ks}, rank_nonzero)


def save_singular_values(activations: torch.Tensor, path: str) -> None:
    A = activations.detach().float().cpu().numpy()
    A = A - A.mean(axis=0)
    sigma = np.linalg.svd(A, full_matrices=False, compute_uv=False)
    np.save(path, sigma)


def same_sum_clustering(
    resid: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float, float, int]:
    """
    Compute (D_within, D_between, ratio, num_examples).

    D_within  = (1/N) Σ_i ||z_i - μ_{c_i}||²
    D_between = (1/(C(C-1))) Σ_{c != c'} ||μ_c - μ_{c'}||²

    Distances are squared L2 (per doc section 12).
    """
    A = resid.detach().float().cpu().numpy()
    y = labels.detach().cpu().numpy()
    classes = np.unique(y)
    # Skip classes with < 1 example (all should have ≥1 by construction).
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
    # Each off-diagonal pair appears once in iu; doc denominator is C*(C-1).
    # The numerator is sum over ordered pairs (c != c'); upper-triangle sum × 2.
    D_between = float(sq[iu].sum() * 2 / (C * (C - 1)))
    ratio = D_within / D_between if D_between > 0 else float("nan")
    return (D_within, D_between, ratio, n_total)


def extract_activations_with_labels(
    model: BlockScheduleTransformer,
    loader,
    pred_position: int,
    n_examples: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Wrap model.extract_residual_activations() and also return the matching
    labels in the same example order.
    """
    activations = model.extract_residual_activations(
        loader, pred_position=pred_position, n_examples=n_examples, device=device,
    )
    collected_labels: list[torch.Tensor] = []
    n = 0
    for _inputs, lab in loader:
        collected_labels.append(lab)
        n += lab.size(0)
        if n >= n_examples:
            break
    labels = torch.cat(collected_labels, dim=0)[:n_examples]
    return activations, labels


def componentwise_erank_rows(
    activations: dict[str, torch.Tensor],
    schedule: list[str],
    *,
    run_id: str,
    depth: int,
    seed: int,
    checkpoint_label: str,
    epoch: int,
    train_acc: float | None,
    val_acc: float | None,
    test_acc_at_best_val: float | None,
) -> list[dict]:
    """
    Produce one row per block following the doc-required schema.

    For a block of kind 'A' or 'M':
      erank_before = eRank(resid_pre)
      erank_after  = eRank(resid_post)
    For 'AM' (combined block): identical to the above, capturing both attention
    and MLP within one block_index. block_type is annotated as the kind string.
    """
    rows: list[dict] = []
    schedule_str = "".join(schedule)
    for i, kind in enumerate(schedule):
        pre = activations[f"blocks.{i}.hook_resid_pre"]
        post = activations[f"blocks.{i}.hook_resid_post"]
        e_before = compute_erank(pre)
        e_after = compute_erank(post)
        delta = e_after - e_before
        rel = (delta / e_before) if e_before > 0 else float("nan")
        rows.append({
            "run_id": run_id,
            "depth": depth,
            "schedule": schedule_str,
            "seed": seed,
            "checkpoint_label": checkpoint_label,
            "epoch": epoch,
            "train_acc": train_acc if train_acc is not None else "",
            "val_acc": val_acc if val_acc is not None else "",
            "test_acc_at_best_val_if_applicable":
                test_acc_at_best_val if test_acc_at_best_val is not None else "",
            "block_index": i,
            "block_type": kind,
            "erank_before": e_before,
            "erank_after": e_after,
            "delta_erank": delta,
            "relative_delta_erank": rel,
            "is_attention": int("A" in kind),
            "is_mlp": int("M" in kind),
        })
    return rows


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    if not rows:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(fieldnames)
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def append_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    if not rows:
        return
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

COMPONENTWISE_FIELDS = [
    "run_id", "depth", "schedule", "seed",
    "checkpoint_label", "epoch", "train_acc", "val_acc",
    "test_acc_at_best_val_if_applicable",
    "block_index", "block_type",
    "erank_before", "erank_after", "delta_erank", "relative_delta_erank",
    "is_attention", "is_mlp",
]

TRAINING_TIME_ERANK_FIELDS = [
    "run_id", "depth", "schedule", "seed",
    "epoch", "relative_epoch_to_tau", "train_acc", "val_acc",
    "checkpoint_label", "final_rep_erank",
    "sum_delta_A", "sum_delta_M", "mean_delta_A", "mean_delta_M",
    "max_positive_delta_block_index", "max_negative_delta_block_index",
]

SAME_SUM_FIELDS = [
    "run_id", "depth", "schedule", "seed",
    "checkpoint_label", "epoch", "train_acc", "val_acc",
    "D_within", "D_between", "within_between_ratio",
    "num_examples", "analysis_set",
]

SPECTRUM_FIELDS = [
    "run_id", "depth", "schedule", "seed",
    "checkpoint_label", "epoch", "train_acc", "val_acc",
    "final_rep_erank", "rank_nonzero",
    "top1_mass", "top5_mass", "top10_mass", "top20_mass",
    "singular_values_path",
]


def _aggregate_training_time_row(
    *, run_id: str, depth: int, schedule_str: str, seed: int,
    epoch: int, tau: int | None, train_acc: float | None, val_acc: float | None,
    checkpoint_label: str, schedule: list[str],
    activations: dict[str, torch.Tensor],
) -> dict:
    """
    For a single eRank snapshot, compute the doc-required summary row.
    Uses per-block deltas to derive sum_delta_A / sum_delta_M etc.
    """
    n_blocks = len(schedule)
    last_idx = n_blocks - 1
    final_rep_erank = compute_erank(activations[f"blocks.{last_idx}.hook_resid_post"])

    deltas: list[tuple[int, str, float]] = []
    for i, kind in enumerate(schedule):
        pre = activations[f"blocks.{i}.hook_resid_pre"]
        post = activations[f"blocks.{i}.hook_resid_post"]
        d = compute_erank(post) - compute_erank(pre)
        deltas.append((i, kind, d))

    a_deltas = [d for _, k, d in deltas if k == "A"]
    m_deltas = [d for _, k, d in deltas if k == "M"]
    sum_dA = float(sum(a_deltas)) if a_deltas else float("nan")
    sum_dM = float(sum(m_deltas)) if m_deltas else float("nan")
    mean_dA = float(np.mean(a_deltas)) if a_deltas else float("nan")
    mean_dM = float(np.mean(m_deltas)) if m_deltas else float("nan")

    max_pos_idx = max(deltas, key=lambda t: t[2])[0]
    max_neg_idx = min(deltas, key=lambda t: t[2])[0]

    rel_to_tau = (epoch - tau) if tau is not None else ""

    return {
        "run_id": run_id,
        "depth": depth,
        "schedule": schedule_str,
        "seed": seed,
        "epoch": epoch,
        "relative_epoch_to_tau": rel_to_tau,
        "train_acc": train_acc if train_acc is not None else "",
        "val_acc": val_acc if val_acc is not None else "",
        "checkpoint_label": checkpoint_label,
        "final_rep_erank": final_rep_erank,
        "sum_delta_A": sum_dA,
        "sum_delta_M": sum_dM,
        "mean_delta_A": mean_dA,
        "mean_delta_M": mean_dM,
        "max_positive_delta_block_index": max_pos_idx,
        "max_negative_delta_block_index": max_neg_idx,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", required=True,
                        help="Block schedule, e.g. 'AAMM' or 'A,A,M,M'")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--output_root", default="outputs/arch_order_sweep_6gpu")
    parser.add_argument("--depth", type=int, default=None,
                        help="Schedule depth (n_A or n_M). Inferred if omitted.")
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--eval_every", type=int, default=EVAL_EVERY)
    parser.add_argument("--periodic_every", type=int, default=PERIODIC_EVERY)
    parser.add_argument("--periodic_erank_every", type=int, default=PERIODIC_ERANK_EVERY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--p", type=int, default=P)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--train_frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val_frac_of_train", type=float, default=VAL_FRAC_OF_TRAIN)
    parser.add_argument("--sustained_early_stop_evals", type=int,
                        default=SUSTAINED_EARLY_STOP_EVALS)
    args = parser.parse_args()

    schedule = parse_schedule(args.schedule)
    schedule_str = "".join(schedule)
    if args.depth is None:
        nA = sum(1 for k in schedule if k == "A")
        nM = sum(1 for k in schedule if k == "M")
        depth = min(nA, nM) if (nA > 0 and nM > 0) else max(nA, nM, len(schedule) // 2)
    else:
        depth = args.depth

    run_dir = os.path.join(PROJECT_ROOT, args.output_root, "runs", args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Spectra go in a shared subdir per the doc.
    spectra_dir = os.path.join(PROJECT_ROOT, args.output_root, "spectra")
    os.makedirs(spectra_dir, exist_ok=True)

    with open(os.path.join(run_dir, "status.txt"), "w") as f:
        f.write("RUNNING\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.run_name}] device={device}  schedule={schedule}  depth={depth}")

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
    model = BlockScheduleTransformer(
        block_schedule=schedule,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        d_mlp=D_MLP,
        d_vocab=args.p + 1,
        d_vocab_out=args.p,
        n_ctx=3,
        seed=args.seed,
    ).to(device)
    n_params = sum(t.numel() for t in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Save random-init checkpoint first.
    random_init_path = os.path.join(run_dir, "random_init_checkpoint.pt")
    save_checkpoint(random_init_path, model, optimizer, epoch=0)

    config_used = {
        "task": "modular_addition",
        "modulus": args.p,
        "seed": args.seed,
        "block_schedule": schedule,
        "schedule_str": schedule_str,
        "depth": depth,
        "n_layers": len(schedule),
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
        "periodic_every": args.periodic_every,
        "periodic_erank_every": args.periodic_erank_every,
        "train_frac": args.train_frac,
        "val_frac_of_train": args.val_frac_of_train,
        "n_total": split_meta["n_total"],
        "n_train": split_meta["n_train"],
        "n_val": split_meta["n_val"],
        "n_test": split_meta["n_test"],
        "n_params": n_params,
        "run_name": args.run_name,
        "sustained_early_stop_evals": args.sustained_early_stop_evals,
    }
    with open(os.path.join(run_dir, "config_used.yaml"), "w") as f:
        yaml.dump(config_used, f, default_flow_style=False)

    # --- Training loop ---
    history: dict = {
        "epoch": [], "step": [],
        "train_loss": [], "val_loss": [], "test_loss": [],
        "train_acc": [], "val_acc": [], "test_acc": [],
        "best_val_acc_so_far": [], "elapsed_time_seconds": [],
    }
    erank_time_rows: list[dict] = []
    checkpoint_manifest: list[dict] = []

    best_val_acc = -1.0
    best_val_epoch: int | None = None
    best_val_path = os.path.join(run_dir, "best_val_checkpoint.pt")
    final_path = os.path.join(run_dir, "final_checkpoint.pt")

    event_thresholds = list(EVENT_VAL_THRESHOLDS)
    event_ckpt_paths: dict[str, tuple[int, str]] = {}  # label -> (epoch, path)
    train_event_done = False
    sustained_high = 0
    grokking_tau: int | None = None

    def _record_ckpt(kind: str, epoch: int, path: str, train_acc=None, val_acc=None,
                     train_loss=None, val_loss=None) -> None:
        checkpoint_manifest.append({
            "run_id": args.run_name,
            "depth": depth,
            "schedule": schedule_str,
            "checkpoint_label": kind,
            "epoch": int(epoch),
            "train_acc": "" if train_acc is None else train_acc,
            "val_acc": "" if val_acc is None else val_acc,
            "train_loss": "" if train_loss is None else train_loss,
            "val_loss": "" if val_loss is None else val_loss,
            "checkpoint_path": path,
        })

    _record_ckpt("random_init", 0, random_init_path)

    t_start = time.time()
    step = 0
    last_epoch = 0
    last_train_loss = float("nan")
    last_train_acc = float("nan")
    last_val_loss = float("nan")
    last_val_acc = float("nan")
    for epoch in tqdm(range(1, args.max_epochs + 1), desc=f"[{args.run_name}]"):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, criterion)
        step += max(1, len(train_loader))
        last_epoch = epoch
        last_train_loss = train_loss
        last_train_acc = train_acc

        do_eval = (epoch % args.eval_every == 0) or (epoch == 1) or (epoch == args.max_epochs)
        if do_eval:
            val_loss, val_acc = evaluate(model, val_loader, device, criterion)
            test_loss, test_acc = evaluate(model, test_loader, device, criterion)
            last_val_loss = val_loss
            last_val_acc = val_acc
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
                print(
                    f"  ep {epoch:6d} | tr_loss {train_loss:.4f} val_loss {val_loss:.4f} "
                    f"| tr_acc {train_acc:.3f} val_acc {val_acc:.3f} test_acc {test_acc:.3f}"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch
                save_checkpoint(best_val_path, model, optimizer, epoch,
                                extra={"val_acc": val_acc, "val_loss": val_loss})

            # Event ckpts — train_acc >= 0.99 (first hit only)
            if not train_event_done and train_acc >= TRAIN_EVENT_THRESHOLD:
                p_evt = os.path.join(ckpt_dir, f"event_first_train_acc_ge_0.99_ep{epoch}.pt")
                save_checkpoint(p_evt, model, optimizer, epoch,
                                extra={"event": f"train_acc>={TRAIN_EVENT_THRESHOLD}"})
                _record_ckpt("first_train_acc_ge_0.99", epoch, p_evt,
                             train_acc=train_acc, val_acc=val_acc,
                             train_loss=train_loss, val_loss=val_loss)
                event_ckpt_paths["first_train_acc_ge_0.99"] = (epoch, p_evt)
                train_event_done = True

            # Event ckpts — val_acc thresholds (first hit only per threshold)
            for thr in list(event_thresholds):
                if val_acc >= thr:
                    label = f"first_val_acc_ge_{thr:.2f}"
                    p_evt = os.path.join(ckpt_dir, f"event_{label}_ep{epoch}.pt")
                    save_checkpoint(p_evt, model, optimizer, epoch,
                                    extra={"event": f"val_acc>={thr}"})
                    _record_ckpt(label, epoch, p_evt,
                                 train_acc=train_acc, val_acc=val_acc,
                                 train_loss=train_loss, val_loss=val_loss)
                    event_ckpt_paths[label] = (epoch, p_evt)
                    event_thresholds.remove(thr)

            if grokking_tau is None and val_acc >= GROKKING_VAL_THRESHOLD:
                grokking_tau = epoch

            sustained_high = sustained_high + 1 if val_acc >= GROKKING_VAL_THRESHOLD else 0

        # Periodic checkpoint
        if epoch % args.periodic_every == 0:
            p_per = os.path.join(ckpt_dir, f"periodic_ep{epoch}.pt")
            save_checkpoint(p_per, model, optimizer, epoch, extra={"event": "periodic"})
            _record_ckpt("periodic", epoch, p_per,
                         train_acc=last_train_acc, val_acc=last_val_acc,
                         train_loss=last_train_loss, val_loss=last_val_loss)

        # Periodic eRank — summary row only (full per-block CSV is for checkpointed labels).
        if epoch % args.periodic_erank_every == 0:
            try:
                acts = model.extract_residual_activations(
                    val_loader, pred_position=2,
                    n_examples=ERANK_N_EXAMPLES, device=device,
                )
                erank_time_rows.append(_aggregate_training_time_row(
                    run_id=args.run_name, depth=depth, schedule_str=schedule_str,
                    seed=args.seed, epoch=epoch, tau=grokking_tau,
                    train_acc=last_train_acc, val_acc=last_val_acc,
                    checkpoint_label="periodic", schedule=schedule, activations=acts,
                ))
            except Exception as exc:
                print(f"  [periodic eRank @ {epoch}] WARNING: {exc}")

        if sustained_high >= args.sustained_early_stop_evals:
            print(f"\n[{args.run_name}] Early stop @ ep {epoch} — val_acc >= "
                  f"{GROKKING_VAL_THRESHOLD} for {args.sustained_early_stop_evals} evals.")
            break

        # Hard-cap stop for schedules that grok and then oscillate around 0.95.
        # If we've ever hit the grokking threshold, don't burn GPU past
        # 2*tau + 1000 epochs hoping for a sustained streak that never lands.
        if grokking_tau is not None and epoch >= 2 * grokking_tau + 1000:
            print(f"\n[{args.run_name}] Hard-cap stop @ ep {epoch} — first grok at "
                  f"{grokking_tau}, hard cap = 2*tau + 1000 = "
                  f"{2 * grokking_tau + 1000}.")
            break

    elapsed = time.time() - t_start

    save_checkpoint(final_path, model, optimizer, last_epoch)
    _record_ckpt("final", last_epoch, final_path,
                 train_acc=last_train_acc, val_acc=last_val_acc,
                 train_loss=last_train_loss, val_loss=last_val_loss)
    if best_val_epoch is not None:
        _record_ckpt("best_val", best_val_epoch, best_val_path,
                     val_acc=best_val_acc)

    # --- Checkpoint manifest CSV ---
    manifest_fields = [
        "run_id", "depth", "schedule", "checkpoint_label", "epoch",
        "train_acc", "val_acc", "train_loss", "val_loss", "checkpoint_path",
    ]
    write_csv(os.path.join(run_dir, "checkpoint_manifest.csv"),
              checkpoint_manifest, manifest_fields)

    # --- Training log CSV ---
    training_log_fields = [
        "run_id", "depth", "schedule", "epoch", "step",
        "train_loss", "val_loss", "test_loss",
        "train_acc", "val_acc", "test_acc",
        "best_val_acc_so_far", "learning_rate", "weight_decay",
        "elapsed_time_seconds",
    ]
    log_rows = []
    for i, ep in enumerate(history["epoch"]):
        log_rows.append({
            "run_id": args.run_name, "depth": depth, "schedule": schedule_str,
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
    write_csv(os.path.join(run_dir, "training_log.csv"), log_rows, training_log_fields)

    # --- Best-val test acc (already in training_log; record explicitly) ---
    best_test_acc = None
    best_test_loss = None
    final_train_acc = history["train_acc"][-1] if history["train_acc"] else float("nan")
    final_val_acc = history["val_acc"][-1] if history["val_acc"] else float("nan")
    final_epoch = last_epoch
    if best_val_epoch is not None and best_val_epoch in history["epoch"]:
        idx = history["epoch"].index(best_val_epoch)
        best_test_acc = history["test_acc"][idx]
        best_test_loss = history["test_loss"][idx]

    # --- Status label ---
    if best_val_acc >= 0.95:
        status = "GROKKED"
    elif final_train_acc >= 0.99 and best_val_acc < 0.20:
        status = "MEMORIZED_ONLY"
    elif final_train_acc < 0.90 and best_val_acc < 0.20:
        status = "FAILED_TO_FIT"
    else:
        status = "PARTIAL"

    # --- Componentwise eRank at key checkpoints ---
    # Helper: load a checkpoint and compute activations + per-block rows.
    def _eval_ckpt(label: str, ckpt_path: str, epoch_for_row: int,
                   train_acc_for_row: float | None,
                   val_acc_for_row: float | None,
                   collect_geom: bool = False,
                   ) -> tuple[list[dict], dict[str, torch.Tensor] | None,
                              torch.Tensor | None]:
        if not os.path.exists(ckpt_path):
            return ([], None, None)
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            m = BlockScheduleTransformer(
                block_schedule=schedule,
                d_model=D_MODEL, n_heads=N_HEADS, d_head=D_HEAD, d_mlp=D_MLP,
                d_vocab=args.p + 1, d_vocab_out=args.p, n_ctx=3, seed=args.seed,
            )
            m.load_state_dict(ckpt["model_state_dict"])
            m = m.to(device)
            if collect_geom:
                acts, labels = extract_activations_with_labels(
                    m, val_loader, pred_position=2,
                    n_examples=ERANK_N_EXAMPLES, device=device,
                )
            else:
                acts = m.extract_residual_activations(
                    val_loader, pred_position=2,
                    n_examples=ERANK_N_EXAMPLES, device=device,
                )
                labels = None
            rows = componentwise_erank_rows(
                acts, schedule,
                run_id=args.run_name, depth=depth, seed=args.seed,
                checkpoint_label=label, epoch=epoch_for_row,
                train_acc=train_acc_for_row, val_acc=val_acc_for_row,
                test_acc_at_best_val=best_test_acc if label == "best_val" else None,
            )
            return (rows, acts, labels)
        except Exception:
            print(f"  [componentwise eRank @ {label}] WARNING")
            traceback.print_exc()
            return ([], None, None)

    # Build the list of (label, path, epoch, train_acc, val_acc) entries.
    def _hist_metrics_at(epoch_val: int) -> tuple[float | None, float | None]:
        if epoch_val in history["epoch"]:
            idx = history["epoch"].index(epoch_val)
            return history["train_acc"][idx], history["val_acc"][idx]
        return (None, None)

    ckpt_labels_for_componentwise: list[tuple[str, str, int, float | None, float | None]] = []
    ckpt_labels_for_componentwise.append(("random_init", random_init_path, 0, None, None))
    if best_val_epoch is not None:
        tr_b, va_b = _hist_metrics_at(best_val_epoch)
        ckpt_labels_for_componentwise.append(
            ("best_val", best_val_path, best_val_epoch, tr_b, va_b)
        )
    tr_f, va_f = _hist_metrics_at(last_epoch)
    ckpt_labels_for_componentwise.append(
        ("final", final_path, last_epoch, tr_f, va_f)
    )
    for label, (epoch_evt, path_evt) in event_ckpt_paths.items():
        tr_e, va_e = _hist_metrics_at(epoch_evt)
        ckpt_labels_for_componentwise.append(
            (label, path_evt, epoch_evt, tr_e, va_e)
        )

    componentwise_all: list[dict] = []
    same_sum_all: list[dict] = []
    spectrum_rows: list[dict] = []
    # For training_time_erank we'll add one row per evaluated checkpoint.
    extra_time_rows: list[dict] = []

    for label, path, ep, tr_a, va_a in ckpt_labels_for_componentwise:
        # Compute same-sum clustering only for labels where it's most informative:
        # random_init, best_val, final, plus the train/val event ckpts if grokked.
        compute_geom = label in ("random_init", "best_val", "final") or (
            status == "GROKKED" and label.startswith("first_val_acc_ge_")
        )
        rows, acts, labels = _eval_ckpt(
            label, path, ep, tr_a, va_a, collect_geom=compute_geom,
        )
        componentwise_all.extend(rows)
        if acts is not None:
            extra_time_rows.append(_aggregate_training_time_row(
                run_id=args.run_name, depth=depth, schedule_str=schedule_str,
                seed=args.seed, epoch=ep, tau=grokking_tau,
                train_acc=tr_a, val_acc=va_a,
                checkpoint_label=label, schedule=schedule, activations=acts,
            ))
        if labels is not None and acts is not None:
            D_w, D_b, ratio, n_ex = same_sum_clustering(
                acts[f"blocks.{len(schedule)-1}.hook_resid_post"], labels,
            )
            same_sum_all.append({
                "run_id": args.run_name, "depth": depth, "schedule": schedule_str,
                "seed": args.seed, "checkpoint_label": label, "epoch": ep,
                "train_acc": "" if tr_a is None else tr_a,
                "val_acc": "" if va_a is None else va_a,
                "D_within": D_w, "D_between": D_b,
                "within_between_ratio": ratio,
                "num_examples": n_ex, "analysis_set": "val",
            })

    # --- Spectrum at best-val ---
    if best_val_epoch is not None and os.path.exists(best_val_path):
        try:
            ckpt = torch.load(best_val_path, map_location="cpu", weights_only=False)
            m_bv = BlockScheduleTransformer(
                block_schedule=schedule,
                d_model=D_MODEL, n_heads=N_HEADS, d_head=D_HEAD, d_mlp=D_MLP,
                d_vocab=args.p + 1, d_vocab_out=args.p, n_ctx=3, seed=args.seed,
            )
            m_bv.load_state_dict(ckpt["model_state_dict"])
            m_bv = m_bv.to(device)
            acts_bv = m_bv.extract_residual_activations(
                val_loader, pred_position=2,
                n_examples=ERANK_N_EXAMPLES, device=device,
            )
            final_resid = acts_bv[f"blocks.{len(schedule)-1}.hook_resid_post"]
            final_e = compute_erank(final_resid)
            mass, rank_nonzero = topk_singular_mass(final_resid, ks=SPECTRUM_KS)
            sv_path = os.path.join(
                spectra_dir, f"{args.run_name}_best_val_singular_values.npy"
            )
            save_singular_values(final_resid, sv_path)
            tr_b, va_b = _hist_metrics_at(best_val_epoch)
            spectrum_rows.append({
                "run_id": args.run_name, "depth": depth, "schedule": schedule_str,
                "seed": args.seed, "checkpoint_label": "best_val", "epoch": best_val_epoch,
                "train_acc": "" if tr_b is None else tr_b,
                "val_acc": "" if va_b is None else va_b,
                "final_rep_erank": final_e, "rank_nonzero": rank_nonzero,
                "top1_mass": mass[1], "top5_mass": mass[5],
                "top10_mass": mass[10], "top20_mass": mass[20],
                "singular_values_path": sv_path,
            })
        except Exception:
            print(f"  [spectrum @ best_val] WARNING")
            traceback.print_exc()

    write_csv(os.path.join(run_dir, "componentwise_erank.csv"),
              componentwise_all, COMPONENTWISE_FIELDS)
    # Merge periodic-only erank_time_rows with extra_time_rows (checkpoint snapshots).
    merged_time = erank_time_rows + extra_time_rows
    merged_time.sort(key=lambda r: (r["epoch"], r["checkpoint_label"]))
    write_csv(os.path.join(run_dir, "training_time_erank.csv"),
              merged_time, TRAINING_TIME_ERANK_FIELDS)
    write_csv(os.path.join(run_dir, "same_sum_clustering.csv"),
              same_sum_all, SAME_SUM_FIELDS)
    write_csv(os.path.join(run_dir, "spectrum_metrics.csv"),
              spectrum_rows, SPECTRUM_FIELDS)

    # --- Plot training curves ---
    try:
        ep_arr = history["epoch"]
        fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(11, 4))
        ax_l.plot(ep_arr, history["train_loss"], label="train")
        ax_l.plot(ep_arr, history["val_loss"], label="val")
        ax_l.plot(ep_arr, history["test_loss"], label="test", linestyle="--", alpha=0.6)
        ax_l.set_xlabel("epoch")
        ax_l.set_ylabel("loss")
        ax_l.set_yscale("log")
        ax_l.set_title(f"{args.run_name} loss")
        ax_l.legend()

        ax_a.plot(ep_arr, history["train_acc"], label="train")
        ax_a.plot(ep_arr, history["val_acc"], label="val")
        ax_a.plot(ep_arr, history["test_acc"], label="test", linestyle="--", alpha=0.6)
        ax_a.axhline(1 / args.p, color="red", linestyle=":", linewidth=1,
                     label=f"chance (1/{args.p})")
        ax_a.set_xlabel("epoch")
        ax_a.set_ylabel("accuracy")
        ax_a.set_ylim(0, 1.05)
        ax_a.set_title(f"{args.run_name} accuracy (sched={schedule_str})")
        ax_a.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "training.png"), dpi=140)
        plt.close(fig)
    except Exception:
        traceback.print_exc()

    # --- Run report ---
    rep_lines = []
    rep_lines.append(f"# Run: {args.run_name}")
    rep_lines.append("")
    rep_lines.append(f"- schedule: `{schedule_str}`  (depth {depth}, {len(schedule)} blocks)")
    rep_lines.append(f"- params: {n_params:,}")
    rep_lines.append(f"- p={args.p}, seed={args.seed}, lr={args.lr}, "
                     f"weight_decay={args.weight_decay}, batch={args.batch_size}")
    rep_lines.append(f"- train/val/test = {split_meta['n_train']}/{split_meta['n_val']}/{split_meta['n_test']}")
    rep_lines.append(f"- ran {last_epoch} epochs in {elapsed/3600:.2f} h "
                     f"(cap {args.max_epochs})")
    rep_lines.append("")
    rep_lines.append("## Best-val checkpoint")
    rep_lines.append(
        f"- best_val_acc = {best_val_acc:.4f} @ epoch {best_val_epoch}  "
        f"(test_acc@best_val = "
        f"{best_test_acc if best_test_acc is None else f'{best_test_acc:.4f}'})"
    )
    rep_lines.append(f"- grokking_tau (first val_acc >= {GROKKING_VAL_THRESHOLD}): "
                     f"{grokking_tau if grokking_tau is not None else 'NA'}")
    rep_lines.append(f"- status: **{status}**")
    rep_lines.append("")
    if componentwise_all:
        rep_lines.append("## Componentwise eRank @ best-val")
        rep_lines.append("")
        rep_lines.append("| block | type | erank_before | erank_after | Δ_erank |")
        rep_lines.append("|---:|:--|---:|---:|---:|")
        for r in componentwise_all:
            if r["checkpoint_label"] != "best_val":
                continue
            rep_lines.append(
                f"| {r['block_index']} | {r['block_type']} | "
                f"{r['erank_before']:.3f} | {r['erank_after']:.3f} | "
                f"{r['delta_erank']:+.3f} |"
            )
        rep_lines.append("")
    if spectrum_rows:
        r = spectrum_rows[0]
        rep_lines.append("## Spectrum @ best-val")
        rep_lines.append(
            f"- final_rep_erank: {r['final_rep_erank']:.3f}, "
            f"rank_nonzero: {r['rank_nonzero']}"
        )
        rep_lines.append(
            f"- top-k mass: 1={r['top1_mass']:.3f}, 5={r['top5_mass']:.3f}, "
            f"10={r['top10_mass']:.3f}, 20={r['top20_mass']:.3f}"
        )
        rep_lines.append("")
    if same_sum_all:
        rep_lines.append("## Same-sum clustering")
        rep_lines.append("")
        rep_lines.append("| ckpt | epoch | D_within | D_between | ratio |")
        rep_lines.append("|:--|---:|---:|---:|---:|")
        for r in same_sum_all:
            rep_lines.append(
                f"| {r['checkpoint_label']} | {r['epoch']} | "
                f"{r['D_within']:.3f} | "
                f"{r['D_between'] if isinstance(r['D_between'], str) else f'{r['D_between']:.3f}'} | "
                f"{r['within_between_ratio'] if isinstance(r['within_between_ratio'], str) else f'{r['within_between_ratio']:.3f}'} |"
            )
        rep_lines.append("")
    rep_lines.append("![training](training.png)")
    with open(os.path.join(run_dir, "run_report.md"), "w") as f:
        f.write("\n".join(rep_lines))

    # --- Summary JSON for the aggregator's per-run pickup ---
    summary = {
        "run_id": args.run_name,
        "schedule": schedule_str,
        "depth": depth,
        "n_blocks": len(schedule),
        "n_params": n_params,
        "best_val_acc": best_val_acc,
        "best_val_epoch": best_val_epoch,
        "test_acc_at_best_val": best_test_acc,
        "test_loss_at_best_val": best_test_loss,
        "final_train_acc": float(final_train_acc),
        "final_val_acc": float(final_val_acc),
        "final_epoch": int(final_epoch),
        "grokking_tau": grokking_tau,
        "status": status,
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(run_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    with open(os.path.join(run_dir, "status.txt"), "w") as f:
        f.write(f"{status}\n")
    print(f"[{args.run_name}] DONE — {status} — outputs in {run_dir}")


if __name__ == "__main__":
    main()
