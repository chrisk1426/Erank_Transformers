"""
analysis/l1_mlp_wu_alignment.py

Test the hypothesis that L1_MLP on the KV task implements a learned soft
projection toward the unembedding subspace:

    H1 (boost):   MLP output's principal directions are aligned with W_U
                  columns, increasing residual-stream variance in the
                  readout subspace.

    H2 (cancel):  MLP output is anti-aligned with noise directions in
                  resid_mid that are orthogonal to W_U, decreasing
                  residual-stream variance in the orthogonal complement.

Method:
  1. Extract resid_mid_L1, resid_post_L1, mlp_out_L1 at the query position
     for 500 KV examples (best-val checkpoint, seeds 0-2).
  2. Build the readout subspace: top-k left singular vectors of W_U where
     k = d_vocab_out. Span of those vectors is exactly the d_vocab_out -
     dimensional subspace that can produce non-zero logits.
  3. Project resid_mid and resid_post into [readout subspace, orthogonal
     complement] and report variance / eRank in each.
  4. Verdict:
        - var(post; readout)  >  var(mid; readout)  -> H1 confirmed
        - var(post; ortho)    <  var(mid; ortho)    -> H2 confirmed
        - eRank should drop in the orthogonal complement specifically.
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
import numpy as np
import torch
import yaml

from analysis.erank import compute_erank
from data.splits import get_key_value_dataloaders_3way
from models.transformer import create_standard_transformer

BASE_DIR = os.path.join(PROJECT_ROOT, "outputs", "thesis_additions")
OUT_DIR  = os.path.join(PROJECT_ROOT, "outputs", "l1_mlp_wu_alignment")
SEEDS    = [0, 1, 2]
N_EXAMPLES = 500


def _extract_l1_components(model, loader, n_examples, device):
    """Collect resid_mid_L1, resid_post_L1, mlp_out_L1 at the query position."""
    names = ["blocks.1.hook_resid_mid",
             "blocks.1.hook_resid_post",
             "blocks.1.hook_mlp_out"]
    bufs = {n: [] for n in names}
    collected = 0
    pos = model.cfg.n_ctx - 1
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            if collected >= n_examples:
                break
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            remaining = n_examples - collected
            if batch_size > remaining:
                inputs = inputs[:remaining]
                batch_size = remaining
            _logits, cache = model.run_with_cache(
                inputs, names_filter=lambda n: n in names, return_type="logits",
            )
            for n in names:
                bufs[n].append(cache[n][:, pos, :].detach().cpu())
            collected += batch_size
    return {n: torch.cat(bufs[n], dim=0)[:n_examples] for n in names}


def _wu_basis(model):
    """
    Return (Q_read, Q_orth) — orthonormal bases of the readout subspace and
    its orthogonal complement in the d_model space.

    W_U has shape (d_model, d_vocab_out).  The column-space of W_U is the
    set of directions the unembedding can read: a vector v in d_model space
    produces logits W_U^T v, which depends only on v's projection onto
    col(W_U).  So col(W_U) is the readout subspace.
    """
    W_U = model.W_U.detach().cpu().numpy()      # (d_model, d_vocab_out)
    d_model, d_vocab_out = W_U.shape
    # Orthonormal basis of col(W_U) via SVD of W_U.
    U, S, _Vt = np.linalg.svd(W_U, full_matrices=True)
    # First d_vocab_out columns of U span col(W_U) (since W_U has rank <= d_vocab_out).
    Q_read = U[:, :d_vocab_out]                  # (d_model, d_vocab_out)
    Q_orth = U[:, d_vocab_out:]                  # (d_model, d_model - d_vocab_out)
    return Q_read, Q_orth, S


def _variance_in_subspace(X: np.ndarray, Q: np.ndarray) -> float:
    """Variance (sum of squared deviations / n) of column-centered X projected onto Q."""
    Xc = X - X.mean(axis=0, keepdims=True)
    proj = Xc @ Q                                # (n, k)
    return float((proj ** 2).sum() / X.shape[0])


def _erank_in_subspace(X: np.ndarray, Q: np.ndarray) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    proj = Xc @ Q                                # (n, k)
    return compute_erank(torch.tensor(proj, dtype=torch.float32))


def _mlp_pc_alignment(mlp_out: np.ndarray, Q_read: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Compute the alignment of MLP output principal directions with the
    readout subspace.

    Returns:
        avg_top_alignment: mean ||Q_read^T pc_i||^2 over the top-k PCs of mlp_out
                           (where k = readout dim), in [0, 1].
        per_pc: array of ||Q_read^T pc_i||^2 for the top-k PCs.
    """
    Mc = mlp_out - mlp_out.mean(axis=0, keepdims=True)
    # PCs of mlp_out are the right singular vectors (column directions).
    _U, _S, Vt = np.linalg.svd(Mc, full_matrices=False)   # Vt: (k, d_model)
    pcs = Vt.T                                            # (d_model, k)  k=min(n,d)
    k = Q_read.shape[1]
    top_pcs = pcs[:, :k]                                  # (d_model, k)
    proj = Q_read.T @ top_pcs                             # (k_read, k_pc)
    per_pc = (proj ** 2).sum(axis=0)                      # (k_pc,)
    return float(per_pc.mean()), per_pc


def run_one(seed, cfg, device):
    ckpt = os.path.join(BASE_DIR, f"seed_{seed}", "standard_key_value",
                        "checkpoints", "model_best_val.pt")
    if not os.path.exists(ckpt):
        print(f"  [SKIP seed {seed}: no checkpoint]")
        return None
    print(f"\n  seed {seed}")
    overrides = {"kv_num_keys": cfg["kv_num_keys"],
                 "kv_vocab_size": cfg["kv_vocab_size"]}
    model = create_standard_transformer("key_value", cfg_overrides=overrides, seed=seed)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device).eval()

    _, _, test_loader, _ = get_key_value_dataloaders_3way(
        num_keys=cfg["kv_num_keys"], vocab_size=cfg["kv_vocab_size"],
        train_frac=cfg["kv_train_frac"], batch_size=cfg["batch_size"], seed=seed,
    )

    acts = _extract_l1_components(model, test_loader, N_EXAMPLES, device)
    Q_read, Q_orth, S_wu = _wu_basis(model)
    d_model = Q_read.shape[0]
    k_read  = Q_read.shape[1]

    mid_np  = acts["blocks.1.hook_resid_mid"].float().numpy()
    post_np = acts["blocks.1.hook_resid_post"].float().numpy()
    mlp_np  = acts["blocks.1.hook_mlp_out"].float().numpy()

    var_mid_read  = _variance_in_subspace(mid_np,  Q_read)
    var_post_read = _variance_in_subspace(post_np, Q_read)
    var_mid_orth  = _variance_in_subspace(mid_np,  Q_orth)
    var_post_orth = _variance_in_subspace(post_np, Q_orth)
    er_mid_read   = _erank_in_subspace(mid_np,  Q_read)
    er_post_read  = _erank_in_subspace(post_np, Q_read)
    er_mid_orth   = _erank_in_subspace(mid_np,  Q_orth)
    er_post_orth  = _erank_in_subspace(post_np, Q_orth)
    align_mean, _ = _mlp_pc_alignment(mlp_np, Q_read)

    sanity = abs((var_mid_read + var_mid_orth)
                 - _variance_in_subspace(mid_np, np.eye(d_model, dtype=np.float64))) < 1e-3

    print(f"    d_model={d_model}  k_read={k_read}  d_ortho={d_model - k_read}")
    print(f"    SUBSPACE             var(mid)    var(post)   Δvar       eRank(mid)  eRank(post)  ΔeRank")
    print(f"    readout (W_U col)    {var_mid_read:>10.3f}  {var_post_read:>10.3f}  "
          f"{var_post_read-var_mid_read:+10.3f}  {er_mid_read:>10.3f}  {er_post_read:>10.3f}  "
          f"{er_post_read-er_mid_read:+8.3f}")
    print(f"    orthogonal (noise)   {var_mid_orth:>10.3f}  {var_post_orth:>10.3f}  "
          f"{var_post_orth-var_mid_orth:+10.3f}  {er_mid_orth:>10.3f}  {er_post_orth:>10.3f}  "
          f"{er_post_orth-er_mid_orth:+8.3f}")
    print(f"    mlp top-{k_read}-PC alignment with W_U col-space (mean) = {align_mean:.4f}")
    print(f"    variance decomposition sanity (sums match): {sanity}")

    return {
        "seed": seed,
        "d_model": d_model, "k_read": k_read,
        "var_mid_read": var_mid_read,  "var_post_read": var_post_read,
        "var_mid_orth": var_mid_orth,  "var_post_orth": var_post_orth,
        "delta_var_read": var_post_read - var_mid_read,
        "delta_var_orth": var_post_orth - var_mid_orth,
        "erank_mid_read": er_mid_read, "erank_post_read": er_post_read,
        "erank_mid_orth": er_mid_orth, "erank_post_orth": er_post_orth,
        "delta_erank_read": er_post_read - er_mid_read,
        "delta_erank_orth": er_post_orth - er_mid_orth,
        "mlp_pc_alignment_mean": align_mean,
        "W_U_singular_values": S_wu.tolist(),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(PROJECT_ROOT, "configs", "default.yaml")) as f:
        cfg = yaml.safe_load(f)
    print(f"Device: {device}")

    results = []
    for seed in SEEDS:
        r = run_one(seed, cfg, device)
        if r is not None:
            results.append(r)

    csv_path = os.path.join(OUT_DIR, "l1_mlp_wu_alignment.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "d_model", "k_read",
            "var_mid_read", "var_post_read", "delta_var_read",
            "var_mid_orth", "var_post_orth", "delta_var_orth",
            "erank_mid_read", "erank_post_read", "delta_erank_read",
            "erank_mid_orth", "erank_post_orth", "delta_erank_orth",
            "mlp_pc_alignment_mean",
        ])
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in writer.fieldnames}
            writer.writerow(row)
    print(f"\nSaved: {csv_path}")
    json_path = os.path.join(OUT_DIR, "l1_mlp_wu_alignment_full.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")

    # Summary across seeds
    def m(key): return float(np.mean([r[key] for r in results]))
    def s(key): return float(np.std([r[key] for r in results]))
    print("\n" + "=" * 70)
    print("Mean ± std across seeds")
    print("=" * 70)
    print(f"  Δvar  in readout subspace:   {m('delta_var_read'):+.3f} ± {s('delta_var_read'):.3f}")
    print(f"  Δvar  in orthogonal noise:   {m('delta_var_orth'):+.3f} ± {s('delta_var_orth'):.3f}")
    print(f"  ΔeRank in readout subspace:  {m('delta_erank_read'):+.3f} ± {s('delta_erank_read'):.3f}")
    print(f"  ΔeRank in orthogonal noise:  {m('delta_erank_orth'):+.3f} ± {s('delta_erank_orth'):.3f}")
    print(f"  MLP top-PC alignment w/ W_U: {m('mlp_pc_alignment_mean'):.4f} "
          f"± {s('mlp_pc_alignment_mean'):.4f}")
    print(f"  (Random baseline alignment:  k_read/d_model = "
          f"{results[0]['k_read']/results[0]['d_model']:.4f})")

    _make_plot(results)


def _make_plot(results):
    seeds = [r["seed"] for r in results]
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Variance panel
    ax = axes[0]
    x = np.arange(len(seeds))
    mid_read  = [r["var_mid_read"]  for r in results]
    post_read = [r["var_post_read"] for r in results]
    mid_orth  = [r["var_mid_orth"]  for r in results]
    post_orth = [r["var_post_orth"] for r in results]
    ax.bar(x - 1.5*width, mid_read,  width, label="mid · readout",  color="#1976D2")
    ax.bar(x - 0.5*width, post_read, width, label="post · readout", color="#42A5F5")
    ax.bar(x + 0.5*width, mid_orth,  width, label="mid · orthogonal",  color="#D32F2F")
    ax.bar(x + 1.5*width, post_orth, width, label="post · orthogonal", color="#EF9A9A")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Variance (sum sq dev / n)")
    ax.set_title("Variance before/after L1_MLP\nsplit by W_U-aligned vs orthogonal subspace")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Δ panel
    ax = axes[1]
    d_var_read = [r["delta_var_read"] for r in results]
    d_var_orth = [r["delta_var_orth"] for r in results]
    ax.bar(x - 0.5*width, d_var_read, width, label="Δvar · readout",     color="#1976D2")
    ax.bar(x + 0.5*width, d_var_orth, width, label="Δvar · orthogonal",  color="#D32F2F")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Δvariance (post − mid)")
    ax.set_title("L1_MLP effect on variance\n+ = boosting,  − = cancelling")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("KV task — does L1_MLP boost the W_U-aligned subspace and cancel the orthogonal complement?",
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "l1_mlp_wu_variance.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {p}")


if __name__ == "__main__":
    main()
