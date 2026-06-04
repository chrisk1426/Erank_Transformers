"""
analysis/audit_erank_pipeline.py

Sanity checks on the eRank computation pipeline against a real KV checkpoint.

Checks:
  1. compute_erank passes Roy-Vetterli synthetic tests (already in erank.py)
  2. resid_post[L] == resid_pre[L+1] in the extracted activations
  3. Centering does what it claims (mean ~ 0 after centering)
  4. eRank is consistent under column permutation (it should be invariant)
  5. eRank is invariant under orthogonal column rotations (variance-preserving)
  6. eRank is scale-invariant in input (since p = sigma/sum(sigma))
  7. The signs of delta_attn, delta_mlp on KV match the reported values
"""

from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from analysis.erank import compute_erank
from analysis.extract_activations import load_model_and_extract


def check_pipeline_against_kv():
    print("=" * 70)
    print("Audit: eRank pipeline on real KV checkpoint (seed 0)")
    print("=" * 70)

    ckpt = os.path.join(
        PROJECT_ROOT, "outputs", "thesis_additions", "seed_0",
        "standard_key_value", "checkpoints", "model_best_val.pt"
    )
    if not os.path.exists(ckpt):
        print(f"[SKIP] checkpoint not found: {ckpt}")
        return

    acts = load_model_and_extract(
        ckpt, "key_value", "standard",
        n_examples=500,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # ------------------------------------------------------------------
    # CHECK A — resid_post[L] should equal resid_pre[L+1]
    # (TransformerLens invariant — confirms we're slicing the same position
    # consistently and that the cache plumbing is correct)
    # ------------------------------------------------------------------
    print("\n[A] resid_post[L]  vs  resid_pre[L+1]  (should be identical)")
    for L in range(1):  # 2-layer model: only L=0 -> L=1
        post = acts[f"blocks.{L}.hook_resid_post"]
        nxt  = acts[f"blocks.{L+1}.hook_resid_pre"]
        diff = (post - nxt).abs().max().item()
        same = torch.allclose(post, nxt, atol=1e-6)
        print(f"  layer {L}->{L+1}: max|diff|={diff:.2e}  identical={same}")

    # ------------------------------------------------------------------
    # CHECK B — centering: after centering, column means must be ~0
    # ------------------------------------------------------------------
    print("\n[B] centering correctness (max |column mean after centering|)")
    for name in ["blocks.0.hook_resid_pre",
                 "blocks.0.hook_resid_mid",
                 "blocks.0.hook_resid_post",
                 "blocks.1.hook_resid_mid"]:
        A = acts[name].float().numpy()
        Ac = A - A.mean(axis=0)
        print(f"  {name:35s}  raw mean abs={np.abs(A.mean(axis=0)).max():.3e}"
              f"   centered mean abs={np.abs(Ac.mean(axis=0)).max():.3e}")

    # ------------------------------------------------------------------
    # CHECK C — eRank invariances:
    #   (1) permutation of columns: invariant
    #   (2) orthogonal rotation of columns: invariant
    #   (3) global scale of matrix: invariant
    # ------------------------------------------------------------------
    A = acts["blocks.1.hook_resid_mid"]
    er_base = compute_erank(A)
    rng = np.random.default_rng(0)
    perm = rng.permutation(A.shape[1])
    er_perm = compute_erank(A[:, perm])
    Q, _ = np.linalg.qr(rng.standard_normal((A.shape[1], A.shape[1])))
    er_rot = compute_erank(torch.tensor(A.numpy() @ Q.astype(np.float32)))
    er_scl = compute_erank(A * 1e3)
    print("\n[C] invariances on resid_mid layer 1:")
    print(f"   base eRank             = {er_base:.4f}")
    print(f"   column-permuted        = {er_perm:.4f}  Δ={er_perm-er_base:+.2e}")
    print(f"   orthogonal-rotated     = {er_rot:.4f}  Δ={er_rot-er_base:+.2e}")
    print(f"   scaled by 1e3          = {er_scl:.4f}  Δ={er_scl-er_base:+.2e}")

    # ------------------------------------------------------------------
    # CHECK D — bounds: 1 <= eRank <= min(n, d)
    # ------------------------------------------------------------------
    print("\n[D] bounds (1 <= eRank <= min(n_examples, d_model))")
    n_ex, d_model = A.shape
    print(f"   n_examples={n_ex}  d_model={d_model}  upper bound={min(n_ex,d_model)}")
    for name, mat in acts.items():
        er = compute_erank(mat)
        in_bounds = (1.0 - 1e-6) <= er <= (min(mat.shape) + 1e-6)
        flag = "OK" if in_bounds else "OUT OF BOUNDS"
        print(f"   {name:35s} eRank={er:.4f}  [{flag}]")

    # ------------------------------------------------------------------
    # CHECK E — reproduce delta signs vs. reported numbers
    # ------------------------------------------------------------------
    print("\n[E] reproducing reported KV deltas")
    e_pre0  = compute_erank(acts["blocks.0.hook_resid_pre"])
    e_mid0  = compute_erank(acts["blocks.0.hook_resid_mid"])
    e_post0 = compute_erank(acts["blocks.0.hook_resid_post"])
    e_pre1  = compute_erank(acts["blocks.1.hook_resid_pre"])
    e_mid1  = compute_erank(acts["blocks.1.hook_resid_mid"])
    e_post1 = compute_erank(acts["blocks.1.hook_resid_post"])
    print(f"   layer 0: pre={e_pre0:.4f}  mid={e_mid0:.4f}  post={e_post0:.4f}")
    print(f"   layer 1: pre={e_pre1:.4f}  mid={e_mid1:.4f}  post={e_post1:.4f}")
    print(f"   L0 delta_attn = mid - pre  = {e_mid0 - e_pre0:+.4f}")
    print(f"   L0 delta_mlp  = post - mid = {e_post0 - e_mid0:+.4f}")
    print(f"   L1 delta_attn = mid - pre  = {e_mid1 - e_pre1:+.4f}")
    print(f"   L1 delta_mlp  = post - mid = {e_post1 - e_mid1:+.4f}")

    # ------------------------------------------------------------------
    # CHECK F — what does resid_pre L0 actually look like?
    # If it's the embeddings at a fixed token position with mostly identical
    # tokens across examples, the *centered* matrix will be near-rank-1
    # in the position embedding (zero variance), and eRank ~ 1.
    # ------------------------------------------------------------------
    print("\n[F] resid_pre layer 0 — variance structure check")
    A0 = acts["blocks.0.hook_resid_pre"].float().numpy()
    col_var = A0.var(axis=0)
    print(f"   shape={A0.shape}  max col var={col_var.max():.4e}  "
          f"min col var={col_var.min():.4e}")
    Ac = A0 - A0.mean(axis=0)
    sigma = np.linalg.svd(Ac, compute_uv=False)
    print(f"   top-5 singular values: {sigma[:5]}")
    print(f"   sum top-5 / sum all = {sigma[:5].sum()/sigma.sum():.4f}")

    print("\nAudit complete.")


if __name__ == "__main__":
    check_pipeline_against_kv()
