"""
scripts/verify_erank.py

Extended verification of the eRank function on synthetic matrices.

Runs 5 test cases (the 3 from analysis/erank.py plus 2 additional ones)
and generates a rank-linearity plot.

Test cases:
  1. Rank-1 matrix         → eRank ≈ 1
  2. Equal singular values → eRank ≈ 50
  3. Random Gaussian       → eRank > 40
  4. Known-rank-k matrix   → eRank ≈ k  (k=10)
  5. Linearly increasing rank (1, 5, 10, 20, 50) → plot + linearity check

Output: figures/erank_rank_linearity.png
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from analysis.erank import compute_erank


def _make_rank_k_matrix(
    k: int,
    n_rows: int,
    n_cols: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create a matrix of shape (n_rows, n_cols) with exact rank k.
    Rows are random linear combinations of k orthonormal basis vectors.
    """
    basis = rng.standard_normal((k, n_cols))
    # Orthonormalize the basis rows so singular values are controlled.
    basis, _ = np.linalg.qr(basis.T)
    basis = basis.T[:k]  # shape (k, n_cols)
    coeffs = rng.standard_normal((n_rows, k))
    return coeffs @ basis  # shape (n_rows, n_cols), rank = k


def run_verification(figures_dir: str) -> None:
    rng = np.random.default_rng(42)
    all_passed = True

    # ------------------------------------------------------------------
    # Test 1: Rank-1 matrix
    # ------------------------------------------------------------------
    col = rng.standard_normal(50)
    A1 = np.outer(np.ones(1000), col)
    r1 = compute_erank(torch.tensor(A1, dtype=torch.float32))
    ok = abs(r1 - 1.0) < 0.1
    print(f"Test 1 — rank-1 matrix:         eRank = {r1:.4f}  (expected ≈ 1)  {'PASSED' if ok else 'FAILED'}")
    all_passed &= ok

    # ------------------------------------------------------------------
    # Test 2: Equal singular values (eRank = 50)
    # ------------------------------------------------------------------
    U, _ = np.linalg.qr(rng.standard_normal((1000, 50)))
    V, _ = np.linalg.qr(rng.standard_normal((50, 50)))
    A2 = U @ V.T
    r2 = compute_erank(torch.tensor(A2, dtype=torch.float32))
    ok = abs(r2 - 50.0) < 2.0
    print(f"Test 2 — equal singular values: eRank = {r2:.4f}  (expected ≈ 50) {'PASSED' if ok else 'FAILED'}")
    all_passed &= ok

    # ------------------------------------------------------------------
    # Test 3: Random Gaussian (1000 × 50), eRank > 40
    # ------------------------------------------------------------------
    A3 = rng.standard_normal((1000, 50))
    r3 = compute_erank(torch.tensor(A3, dtype=torch.float32))
    ok = r3 > 40.0
    print(f"Test 3 — random Gaussian:       eRank = {r3:.4f}  (expected > 40) {'PASSED' if ok else 'FAILED'}")
    all_passed &= ok

    # ------------------------------------------------------------------
    # Test 4: Known-rank-k matrix  (k=10, 1000 × 50)
    # ------------------------------------------------------------------
    k = 10
    A4 = _make_rank_k_matrix(k, n_rows=1000, n_cols=50, rng=rng)
    r4 = compute_erank(torch.tensor(A4, dtype=torch.float32))
    # eRank won't be exactly k because random coefficients spread the
    # singular values, but it should be in [k-3, k+3].
    ok = abs(r4 - k) < 3.0
    print(f"Test 4 — rank-{k} matrix:           eRank = {r4:.4f}  (expected ≈ {k})  {'PASSED' if ok else 'FAILED'}")
    all_passed &= ok

    # ------------------------------------------------------------------
    # Test 5: Linearly increasing rank — plot
    # ------------------------------------------------------------------
    ranks = [1, 5, 10, 20, 50]
    eranks = []
    for r in ranks:
        A = _make_rank_k_matrix(r, n_rows=1000, n_cols=128, rng=rng)
        er = compute_erank(torch.tensor(A, dtype=torch.float32))
        eranks.append(er)
        print(f"  Rank {r:3d} → eRank = {er:.3f}")

    # Check approximate linearity: eRank should increase monotonically
    # and correlation with true rank should be > 0.99.
    correlation = float(np.corrcoef(ranks, eranks)[0, 1])
    ok = correlation > 0.99
    print(f"Test 5 — rank linearity:        corr(rank, eRank) = {correlation:.4f}  (expected > 0.99)  {'PASSED' if ok else 'FAILED'}")
    all_passed &= ok

    # Plot
    os.makedirs(figures_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ranks, eranks, marker="o", color="#4C72B0")
    ax.plot(ranks, ranks,  linestyle="--", color="gray", label="eRank = true rank")
    ax.set_xlabel("True rank")
    ax.set_ylabel("eRank")
    ax.set_title("eRank vs True Rank (synthetic matrices)")
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(figures_dir, "erank_rank_linearity.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    if all_passed:
        print("All eRank verification tests PASSED.")
    else:
        print("WARNING: One or more eRank verification tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    run_verification(figures_dir)
