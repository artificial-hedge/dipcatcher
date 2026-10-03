"""SYNTHETIC orthogonal (subspace) iteration for top-k eigenspace.

Simultaneous iteration with QR normalization; Ritz values converge to the
top-k eigenvalues of a symmetric matrix — verified against eigh.
"""

from __future__ import annotations

import random

import numpy as np


def orth_iter(A: np.ndarray, k: int, iters: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ritz_values, orthonormal basis of dominant k-subspace)."""
    n = A.shape[0]
    Q = np.eye(n)[:, :k]
    for _ in range(iters):
        Z = A @ Q
        Q, _ = np.linalg.qr(Z)
    ritz = np.diag(Q.T @ A @ Q)
    return ritz, Q


def bench_orth_iter(seed: int = 20261231 + 533) -> dict[str, float]:
    rng = random.Random(seed)
    eig_ok = 0
    sub_ok = 0
    n_trials = 30
    for _ in range(n_trials):
        n = rng.randrange(6, 12)
        k = rng.randrange(2, n // 2)
        M = np.array([[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n)])
        A = M + M.T
        # enforce a spectral gap for stable recovery
        A += np.eye(n) * rng.uniform(0, 0.5)
        ritz, Q = orth_iter(A, k, iters=3000)
        w, V = np.linalg.eigh(A)
        idx = np.argsort(-np.abs(w))[:k]
        ref = np.sort(np.abs(w[idx]))
        eig_ok += int(np.allclose(np.sort(np.abs(ritz)), ref, atol=1e-3))
        V = V[:, idx]
        sub_ok += int(np.allclose(Q @ Q.T, V @ V.T, atol=1e-3))
    return {
        "synthetic_ritz_exact": eig_ok / n_trials,
        "synthetic_subspace_exact": sub_ok / n_trials,
    }
