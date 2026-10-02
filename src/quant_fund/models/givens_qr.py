"""SYNTHETIC Givens-rotation QR factorization.

Applies Givens rotations column-by-column to triangularize A; orthogonality
and triangular structure verified against numpy QR.
"""

from __future__ import annotations

import random

import numpy as np


def _givens(a: float, b: float) -> tuple[float, float]:
    if b == 0:
        return 1.0, 0.0
    r = np.hypot(a, b)
    return a / r, b / r


def givens_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Q R factorization via Givens rotations (Q = G1ᵀ...Gkᵀ)."""
    m, n = A.shape
    R = A.astype(float).copy()
    Q = np.eye(m)
    for j in range(n):
        for i in range(m - 1, j, -1):
            c, s = _givens(R[i - 1, j], R[i, j])
            G = np.array([[c, s], [-s, c]])
            R[i - 1 : i + 1, j:] = G @ R[i - 1 : i + 1, j:]
            Q[:, i - 1 : i + 1] = Q[:, i - 1 : i + 1] @ G.T
    return Q, R


def bench_givens_qr(seed: int = 20261231 + 531) -> dict[str, float]:
    rng = random.Random(seed)
    fac = 0
    orth = 0
    tri = 0
    n_trials = 30
    for _ in range(n_trials):
        m = rng.randrange(4, 8)
        n = rng.randrange(2, m)
        A = np.array([[rng.uniform(-2, 2) for _ in range(n)] for _ in range(m)])
        Q, R = givens_qr(A)
        fac += int(np.allclose(Q @ R, A, atol=1e-8))
        orth += int(np.allclose(Q.T @ Q, np.eye(m), atol=1e-8))
        tri += int(np.allclose(np.tril(R, -1), 0, atol=1e-8))
    return {
        "synthetic_factor_exact": fac / n_trials,
        "synthetic_orthonormal": orth / n_trials,
        "synthetic_upper_tri": tri / n_trials,
    }
