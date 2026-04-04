"""
analysis/erank.py

Effective rank (eRank) computation for erank-interp.

Implements the Roy & Vetterli (2007) definition:
  eRank(A) = exp(H(σ̄))
where σ̄ is the normalised singular value distribution of A and H is the
Shannon entropy computed with the natural logarithm.
"""

from __future__ import annotations

import numpy as np
import torch


def compute_erank(activations: torch.Tensor) -> float:
    """
    Compute the effective rank of an activation matrix.

    The matrix is column-centred before the SVD so that eRank reflects the
    variance structure rather than mean offsets (analogous to PCA).

    Args:
        activations: Tensor of shape (n_examples, d_model).
                     Rows = examples, columns = model dimensions.

    Returns:
        erank: float — the effective rank of the activation matrix.
    """
    # Move to CPU and convert to float32 numpy array.
    A: np.ndarray = activations.detach().float().cpu().numpy()

    # Center each column (subtract column mean).
    A = A - A.mean(axis=0)

    # Economy SVD — sigma has shape (min(n_examples, d_model),).
    # We only need the singular values.
    sigma: np.ndarray = np.linalg.svd(A, full_matrices=False, compute_uv=False)

    # Filter out numerically zero singular values.
    # By convention 0 * log(0) = 0, so these contribute nothing to entropy.
    sigma = sigma[sigma > 0]

    if sigma.size == 0:
        # Degenerate case: zero matrix has eRank = 0.
        return 0.0

    # Normalise to a probability distribution over singular values.
    p = sigma / sigma.sum()

    # Shannon entropy with natural log. The 1e-10 epsilon is a safety guard;
    # filtering above makes it largely redundant.
    H: float = float(-np.sum(p * np.log(p + 1e-10)))

    return float(np.exp(H))


def verify_erank() -> None:
    """
    Verify eRank on three synthetic matrices that bracket expected behaviour.

    Test 1 — Rank-1 matrix:        eRank should be ≈ 1
    Test 2 — Equal singular values: eRank should be ≈ 50
    Test 3 — Random Gaussian:       eRank should be > 40
    """
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Test 1: rank-1 matrix (all rows are the same non-zero vector).
    # Only one singular value is non-zero, so H = 0 and eRank = exp(0) = 1.
    # ------------------------------------------------------------------
    col = rng.standard_normal(50)
    A1 = np.outer(np.ones(1000), col)          # shape (1000, 50), rank 1
    result1 = compute_erank(torch.tensor(A1, dtype=torch.float32))
    print(f"Test 1 — rank-1 matrix:         eRank = {result1:.4f}  (expected ≈ 1)")
    assert abs(result1 - 1.0) < 0.1, f"Test 1 FAILED: eRank = {result1:.4f}"
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 2: matrix with 50 equal singular values.
    # Build A = U @ V^T where U (1000×50) and V (50×50) are orthonormal.
    # All 50 singular values equal 1, so H = log(50) and eRank = 50.
    # ------------------------------------------------------------------
    U, _ = np.linalg.qr(rng.standard_normal((1000, 50)))   # (1000, 50), orthonormal cols
    V, _ = np.linalg.qr(rng.standard_normal((50, 50)))     # (50, 50), orthonormal
    A2 = U @ V.T                                            # shape (1000, 50), all σ = 1
    result2 = compute_erank(torch.tensor(A2, dtype=torch.float32))
    print(f"Test 2 — equal singular values: eRank = {result2:.4f}  (expected ≈ 50)")
    assert abs(result2 - 50.0) < 2.0, f"Test 2 FAILED: eRank = {result2:.4f}"
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 3: random Gaussian matrix (1000, 50).
    # Singular values of a Gaussian matrix follow the Marchenko–Pastur law
    # and are approximately equal when n >> d, so eRank should be close to 50.
    # ------------------------------------------------------------------
    A3 = rng.standard_normal((1000, 50))
    result3 = compute_erank(torch.tensor(A3, dtype=torch.float32))
    print(f"Test 3 — random Gaussian:       eRank = {result3:.4f}  (expected > 40)")
    assert result3 > 40.0, f"Test 3 FAILED: eRank = {result3:.4f}"
    print("  PASSED")

    print("\nAll eRank verification tests passed.")


if __name__ == "__main__":
    verify_erank()
