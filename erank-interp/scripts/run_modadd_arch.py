"""
scripts/run_modadd_arch.py

Generic launcher for the Part-A modular-addition architecture-order matrix.

One Python script that handles all 7 schedules; the schedule and run name are
passed via CLI. Trains a BlockScheduleTransformer with validation-based
checkpoint selection (test set is *only* evaluated for diagnostics, never used
for stopping or selection).

Outputs:
    outputs/modadd_arch_matrix_7gpu/<run_name>/
        config_used.yaml
        training_log.csv
        random_init_checkpoint.pt
        best_val_checkpoint.pt
        final_checkpoint.pt
        checkpoints/        (periodic + event)
        checkpoint_manifest.csv
        endpoint_erank.csv
        endpoint_erank.json
        training_time_erank.csv         # if any periodic eRank computed
        spectrum_metrics.csv            # at best-val
        same_sum_clustering.csv         # at best-val (only if val_acc >= 0.5)
        training.png
        status.txt
        run_report_for_chatgpt.md
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

from analysis.compute_erank_profiles import compute_erank_profile
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
PERIODIC_EVERY = 500           # checkpoint every 500 epochs
PERIODIC_ERANK_EVERY = 1000    # eRank every 1000 epochs (lighter)
EVAL_EVERY = 25                # eval val/test every N epochs
EVENT_VAL_THRESHOLDS = [0.05, 0.25, 0.50, 0.75, 0.95]
TRAIN_EVENT_THRESHOLD = 0.99
ERANK_N_EXAMPLES = 500
GROKKING_VAL_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_schedule(s: str) -> list[str]:
    """Parse 'AM,A,M,M' or 'AMAMAMAM' or 'AM-AM-AM-AM' into a block list."""
    s = s.strip()
    if "," in s:
        return [tok.strip() for tok in s.split(",")]
    if "-" in s:
        return [tok.strip() for tok in s.split("-")]
    # tokenise greedy: AM > A > M
    out: list[str] = []
    i = 0
    while i < len(s):
        if s[i:i + 2] == "AM":
            out.append("AM")
            i += 2
        elif s[i] in ("A", "M"):
            out.append(s[i])
            i += 1
        else:
            raise ValueError(f"Bad schedule token at position {i}: {s!r}")
    return out


def evaluate(model, loader, device, criterion) -> tuple[float, float]:
    """Return (avg_loss, accuracy) on a dataloader. Predicts at sequence position 2."""
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)[:, 2, :]  # modular addition prediction position
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


def topk_singular_mass(activations: torch.Tensor, ks: list[int]) -> dict[int, float]:
    """Return {k: top-k singular mass} for the centred activation matrix."""
    A = activations.detach().float().cpu().numpy()
    A = A - A.mean(axis=0)
    sigma = np.linalg.svd(A, full_matrices=False, compute_uv=False)
    total = float(sigma.sum())
    if total == 0:
        return {k: 0.0 for k in ks}
    out: dict[int, float] = {}
    for k in ks:
        out[k] = float(sigma[:k].sum() / total)
    return out


def same_sum_clustering_ratio(
    final_resid: torch.Tensor, labels: torch.Tensor, p: int
) -> float | None:
    """
    Compute ratio of mean within-class to mean between-class L2 distance,
    where class = (a + b) mod p. Returns None if any class has < 2 examples.
    """
    A = final_resid.detach().float().cpu().numpy()
    y = labels.detach().cpu().numpy()
    classes = np.unique(y)
    # Use centroids as a cheap proxy for between-class distance.
    centroids = np.zeros((classes.size, A.shape[1]), dtype=np.float32)
    within: list[float] = []
    for ci, c in enumerate(classes):
        mask = (y == c)
        rows = A[mask]
        if rows.shape[0] < 2:
            continue
        centroid = rows.mean(axis=0)
        centroids[ci] = centroid
        d = np.linalg.norm(rows - centroid, axis=1).mean()
        within.append(float(d))
    if not within:
        return None
    # Between-class: mean pairwise centroid distance (excluding self).
    n = centroids.shape[0]
    if n < 2:
        return None
    diffs = centroids[:, None, :] - centroids[None, :, :]
    pairwise = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(n, k=1)
    between = float(pairwise[iu].mean())
    if between == 0:
        return None
    return float(np.mean(within) / between)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", required=True,
                        help="Block schedule, e.g. 'AM,AM,AM,AM' or 'AMMM' or 'A-A-A-A'")
    parser.add_argument("--run_name", required=True,
                        help="Run name (folder under outputs/modadd_arch_matrix_7gpu/)")
    parser.add_argument("--output_root", default="outputs/modadd_arch_matrix_7gpu")
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
    args = parser.parse_args()

    schedule = parse_schedule(args.schedule)
    out_dir = os.path.join(PROJECT_ROOT, args.output_root, args.run_name)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(out_dir, "status.txt"), "w") as f:
        f.write("RUNNING\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.run_name}] device={device}  schedule={schedule}")

    # Data ---------------------------------------------------------------
    train_loader, val_loader, test_loader, split_meta = get_modular_addition_dataloaders_3way(
        p=args.p,
        train_frac=args.train_frac,
        val_frac_of_train=args.val_frac_of_train,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    with open(os.path.join(out_dir, "split_metadata.json"), "w") as f:
        json.dump(split_meta, f, indent=2)
    print(f"[{args.run_name}] split: {split_meta}")

    # Model --------------------------------------------------------------
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
    print(f"[{args.run_name}] params={n_params:,}")

    # Save random-init checkpoint first thing.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    save_checkpoint(
        os.path.join(out_dir, "random_init_checkpoint.pt"), model, optimizer, epoch=0,
    )

    config_used = {
        "task": "modular_addition",
        "modulus": args.p,
        "seed": args.seed,
        "block_schedule": schedule,
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
    }
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.dump(config_used, f, default_flow_style=False)

    # Training loop ------------------------------------------------------
    history: dict = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
    }
    erank_time_series: list[dict] = []
    checkpoint_manifest: list[dict] = []

    best_val_acc = -1.0
    best_val_epoch: int | None = None
    best_val_path = os.path.join(out_dir, "best_val_checkpoint.pt")
    final_path = os.path.join(out_dir, "final_checkpoint.pt")

    event_thresholds = list(EVENT_VAL_THRESHOLDS)
    train_event_done = False
    sustained_high = 0
    sustained_threshold_epochs = 200  # consecutive evals at val >= 0.95 to early-stop
    grokking_tau: int | None = None

    def _record_ckpt(kind: str, epoch: int, path: str, extra: dict | None = None) -> None:
        row = {"kind": kind, "epoch": int(epoch), "path": path}
        if extra:
            row.update(extra)
        checkpoint_manifest.append(row)

    _record_ckpt("random_init", 0, os.path.join(out_dir, "random_init_checkpoint.pt"))

    t_start = time.time()
    for epoch in tqdm(range(1, args.max_epochs + 1), desc=f"[{args.run_name}]"):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, criterion)

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.max_epochs:
            val_loss, val_acc = evaluate(model, val_loader, device, criterion)
            test_loss, test_acc = evaluate(model, test_loader, device, criterion)
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["test_acc"].append(test_acc)

            if epoch % 500 == 0 or epoch == 1:
                print(
                    f"  ep {epoch:6d} | tr_loss {train_loss:.4f} val_loss {val_loss:.4f} "
                    f"| tr_acc {train_acc:.3f} val_acc {val_acc:.3f} test_acc {test_acc:.3f}"
                )

            # Best-val checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch
                save_checkpoint(best_val_path, model, optimizer, epoch,
                                extra={"val_acc": val_acc, "val_loss": val_loss})

            # Event checkpoints — train_acc >= 0.99
            if not train_event_done and train_acc >= TRAIN_EVENT_THRESHOLD:
                p_evt = os.path.join(ckpt_dir, f"event_train_acc_99_ep{epoch}.pt")
                save_checkpoint(p_evt, model, optimizer, epoch,
                                extra={"event": f"train_acc>={TRAIN_EVENT_THRESHOLD}"})
                _record_ckpt("event_train_acc_99", epoch, p_evt)
                train_event_done = True

            # Event checkpoints — val_acc thresholds
            for thr in list(event_thresholds):
                if val_acc >= thr:
                    p_evt = os.path.join(
                        ckpt_dir, f"event_val_acc_{int(thr * 100):02d}_ep{epoch}.pt"
                    )
                    save_checkpoint(p_evt, model, optimizer, epoch,
                                    extra={"event": f"val_acc>={thr}"})
                    _record_ckpt(f"event_val_acc_{int(thr * 100):02d}", epoch, p_evt)
                    event_thresholds.remove(thr)

            # Grokking time
            if grokking_tau is None and val_acc >= GROKKING_VAL_THRESHOLD:
                grokking_tau = epoch

            # Sustained-high tracking for early stop
            if val_acc >= GROKKING_VAL_THRESHOLD:
                sustained_high += 1
            else:
                sustained_high = 0

        # Periodic checkpoint
        if epoch % args.periodic_every == 0:
            p_per = os.path.join(ckpt_dir, f"periodic_ep{epoch}.pt")
            save_checkpoint(p_per, model, optimizer, epoch, extra={"event": "periodic"})
            _record_ckpt("periodic", epoch, p_per)

        # Periodic eRank
        if epoch % args.periodic_erank_every == 0:
            try:
                acts = model.extract_residual_activations(
                    val_loader, pred_position=2,
                    n_examples=ERANK_N_EXAMPLES, device=device,
                )
                profile = compute_erank_profile(acts, n_layers=len(schedule))
                entry: dict = {"epoch": epoch}
                for li in range(len(schedule)):
                    lk = f"layer_{li}"
                    entry[f"layer_{li}_pre"] = profile["erank"][lk]["pre"]
                    entry[f"layer_{li}_mid"] = profile["erank"][lk]["mid"]
                    entry[f"layer_{li}_post"] = profile["erank"][lk]["post"]
                    entry[f"layer_{li}_delta_attn"] = profile["delta_erank_attn"][lk]
                    entry[f"layer_{li}_delta_mlp"] = profile["delta_erank_mlp"][lk]
                erank_time_series.append(entry)
            except Exception as exc:
                print(f"  [periodic eRank @ {epoch}] WARNING: {exc}")

        # Early stop if val sustained
        if sustained_high >= sustained_threshold_epochs:
            print(f"\n[{args.run_name}] Early stop @ epoch {epoch} — val_acc >= {GROKKING_VAL_THRESHOLD} for {sustained_threshold_epochs} consecutive evals.")
            break

    elapsed = time.time() - t_start

    # Final checkpoint
    last_epoch = history["epoch"][-1] if history["epoch"] else 0
    save_checkpoint(final_path, model, optimizer, last_epoch)
    _record_ckpt("final", last_epoch, final_path)
    if best_val_epoch is not None:
        _record_ckpt("best_val", best_val_epoch, best_val_path,
                     extra={"val_acc": best_val_acc})

    # Manifest CSV
    with open(os.path.join(out_dir, "checkpoint_manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "epoch", "path", "extra"])
        for r in checkpoint_manifest:
            w.writerow([
                r["kind"], r["epoch"], r["path"],
                json.dumps({k: v for k, v in r.items() if k not in ("kind", "epoch", "path")}),
            ])

    # Training log CSV
    with open(os.path.join(out_dir, "training_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "test_loss",
                    "train_acc", "val_acc", "test_acc"])
        for i, ep in enumerate(history["epoch"]):
            w.writerow([
                ep, history["train_loss"][i], history["val_loss"][i],
                history["test_loss"][i], history["train_acc"][i],
                history["val_acc"][i], history["test_acc"][i],
            ])

    # training_time eRank CSV
    if erank_time_series:
        keys = list(erank_time_series[0].keys())
        with open(os.path.join(out_dir, "training_time_erank.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in erank_time_series:
                w.writerow(r)

    # Best-val test acc
    best_test_acc = None
    if best_val_epoch is not None and best_val_epoch in history["epoch"]:
        idx = history["epoch"].index(best_val_epoch)
        best_test_acc = history["test_acc"][idx]

    # Endpoint eRank @ best-val (load from disk so the on-disk weights are what's analysed)
    endpoint_done = False
    endpoint_layer_rows: list[dict] = []
    sum_da = sum_dm = None
    spectrum_rows: list[dict] = []
    cluster_ratio: float | None = None
    final_resid_for_geom = None
    final_labels_for_geom = None

    try:
        ckpt = torch.load(best_val_path, map_location="cpu", weights_only=False)
        eval_model = BlockScheduleTransformer(
            block_schedule=schedule,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            d_head=D_HEAD,
            d_mlp=D_MLP,
            d_vocab=args.p + 1,
            d_vocab_out=args.p,
            n_ctx=3,
            seed=args.seed,
        )
        eval_model.load_state_dict(ckpt["model_state_dict"])
        eval_model = eval_model.to(device)
        acts = eval_model.extract_residual_activations(
            val_loader, pred_position=2, n_examples=ERANK_N_EXAMPLES, device=device,
        )
        profile = compute_erank_profile(acts, n_layers=len(schedule))
        sum_da = float(sum(profile["delta_erank_attn"].values()))
        sum_dm = float(sum(profile["delta_erank_mlp"].values()))
        for li in range(len(schedule)):
            lk = f"layer_{li}"
            kind = schedule[li]
            row = {
                "layer": li,
                "block_kind": kind,
                "erank_pre": profile["erank"][lk]["pre"],
                "erank_mid": profile["erank"][lk]["mid"],
                "erank_post": profile["erank"][lk]["post"],
                "delta_attn": profile["delta_erank_attn"][lk] if "A" in kind else "NA",
                "delta_mlp":  profile["delta_erank_mlp"][lk] if "M" in kind else "NA",
            }
            endpoint_layer_rows.append(row)
        with open(os.path.join(out_dir, "endpoint_erank.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "block_kind", "erank_pre", "erank_mid", "erank_post",
                        "delta_attn", "delta_mlp"])
            for r in endpoint_layer_rows:
                w.writerow([r["layer"], r["block_kind"], r["erank_pre"], r["erank_mid"],
                            r["erank_post"], r["delta_attn"], r["delta_mlp"]])
            w.writerow([])
            w.writerow(["sum_delta_attn", sum_da])
            w.writerow(["sum_delta_mlp", sum_dm])
        with open(os.path.join(out_dir, "endpoint_erank.json"), "w") as f:
            json.dump(profile, f, indent=2, default=float)

        # Spectrum (top-k mass) on the final residual representation R_final.
        last_layer = len(schedule) - 1
        final_resid = acts[f"blocks.{last_layer}.hook_resid_post"]  # (n_examples, d_model)
        spec = topk_singular_mass(final_resid, ks=[1, 5, 10, 20])
        for k, mass in spec.items():
            spectrum_rows.append({"k": k, "topk_mass": mass})
        with open(os.path.join(out_dir, "spectrum_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["k", "topk_mass"])
            w.writeheader()
            for r in spectrum_rows:
                w.writerow(r)

        # Same-sum clustering: only meaningful for grokked runs.
        # Pull labels from the val_loader in the same order extraction used.
        if best_val_acc >= 0.50:
            collected_labels: list[torch.Tensor] = []
            n_needed = ERANK_N_EXAMPLES
            for _inputs, lab in val_loader:
                collected_labels.append(lab)
                if sum(t.numel() for t in collected_labels) >= n_needed:
                    break
            labels = torch.cat(collected_labels, dim=0)[:n_needed]
            final_resid_for_geom = final_resid[:labels.shape[0]]
            final_labels_for_geom = labels
            cluster_ratio = same_sum_clustering_ratio(
                final_resid_for_geom, final_labels_for_geom, p=args.p,
            )
            with open(os.path.join(out_dir, "same_sum_clustering.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["metric", "value"])
                w.writerow(["within_over_between_ratio",
                            cluster_ratio if cluster_ratio is not None else "NA"])
        endpoint_done = True
    except Exception:
        print(f"[{args.run_name}] endpoint eRank failed:")
        traceback.print_exc()

    # Plots --------------------------------------------------------------
    try:
        epochs_arr = history["epoch"]
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
        ax_loss.plot(epochs_arr, history["train_loss"], label="train")
        ax_loss.plot(epochs_arr, history["val_loss"], label="val")
        ax_loss.plot(epochs_arr, history["test_loss"], label="test", linestyle="--", alpha=0.6)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_yscale("log")
        ax_loss.set_title(f"{args.run_name}  loss")
        ax_loss.legend()

        ax_acc.plot(epochs_arr, history["train_acc"], label="train")
        ax_acc.plot(epochs_arr, history["val_acc"], label="val")
        ax_acc.plot(epochs_arr, history["test_acc"], label="test", linestyle="--", alpha=0.6)
        ax_acc.axhline(1 / args.p, color="red", linestyle=":", linewidth=1, label=f"chance (1/{args.p})")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.set_title(f"{args.run_name}  accuracy (sched={'-'.join(schedule)})")
        ax_acc.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "training.png"), dpi=140)
        plt.close(fig)
    except Exception:
        traceback.print_exc()

    # Run report
    report = []
    report.append(f"# Run: {args.run_name}")
    report.append("")
    report.append(f"- schedule: `{schedule}`")
    report.append(f"- params: {n_params:,}")
    report.append(f"- p={args.p}, seed={args.seed}, lr={args.lr}, weight_decay={args.weight_decay}, batch={args.batch_size}")
    report.append(f"- train/val/test = {split_meta['n_train']}/{split_meta['n_val']}/{split_meta['n_test']}")
    report.append(f"- max_epochs (cap): {args.max_epochs};  ran for {last_epoch} epochs in {elapsed/3600:.2f} h")
    report.append("")
    report.append("## Best-val checkpoint")
    report.append(
        f"- best_val_acc = {best_val_acc:.4f} @ epoch {best_val_epoch}  "
        f"(test_acc@best_val = {best_test_acc if best_test_acc is None else f'{best_test_acc:.4f}'})"
    )
    report.append(f"- grokking_tau (first val_acc >= {GROKKING_VAL_THRESHOLD}): {grokking_tau if grokking_tau is not None else 'NA'}")
    report.append("")
    report.append("## Endpoint eRank (best-val)")
    if endpoint_done:
        report.append(f"- sum_delta_attn = {sum_da:.4f},  sum_delta_mlp = {sum_dm:.4f}")
        report.append("")
        report.append("| layer | kind | pre | mid | post | Δ_attn | Δ_mlp |")
        report.append("|---:|:--|---:|---:|---:|---:|---:|")
        for r in endpoint_layer_rows:
            da_s = r["delta_attn"] if isinstance(r["delta_attn"], str) else f"{r['delta_attn']:+.3f}"
            dm_s = r["delta_mlp"] if isinstance(r["delta_mlp"], str) else f"{r['delta_mlp']:+.3f}"
            report.append(
                f"| {r['layer']} | {r['block_kind']} | {r['erank_pre']:.3f} | "
                f"{r['erank_mid']:.3f} | {r['erank_post']:.3f} | {da_s} | {dm_s} |"
            )
        if spectrum_rows:
            report.append("")
            report.append("**Top-k singular mass at final residual:**")
            for r in spectrum_rows:
                report.append(f"- k={r['k']}: {r['topk_mass']:.4f}")
        if cluster_ratio is not None:
            report.append("")
            report.append(f"**Same-sum clustering (within/between):** {cluster_ratio:.4f}")
    else:
        report.append("Endpoint eRank failed; see logs.")
    report.append("")
    report.append("![training](training.png)")
    with open(os.path.join(out_dir, "run_report_for_chatgpt.md"), "w") as f:
        f.write("\n".join(report))

    # Status
    with open(os.path.join(out_dir, "status.txt"), "w") as f:
        f.write("COMPLETED\n")
    print(f"[{args.run_name}] DONE — outputs in {out_dir}")


if __name__ == "__main__":
    main()
