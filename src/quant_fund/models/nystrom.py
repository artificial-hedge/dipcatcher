"""Nyström PSD-approximation canon (Williams & Seeger 2001):
approximate a kernel matrix via landmark columns,
K ≈ C W^+ C^T, evaluated against the exact eigendecomposition
on a synthetic RBF kernel matrix.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def nystrom(
    k_mat: FloatArray,
    landmarks: FloatArray | list[int],
) -> FloatArray:
    """Nyström low-rank approximation of a PSD kernel matrix."""
    k_mat = np.asarray(k_mat, dtype=np.float64)
    idx = np.asarray(landmarks, dtype=np.int64)
    c = k_mat[:, idx]
    w = k_mat[np.ix_(idx, idx)]
    w_pinv = np.linalg.pinv(w)
    return np.asarray(c @ w_pinv @ c.T, dtype=np.float64)


def bench_nystrom(seed: int | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else 20261231)
    n = 500
    x = rng.standard_normal((n, 8))
    d2 = np.sum(x**2, axis=1, keepdims=True) + np.sum(x**2, axis=1)[None, :] - 2.0 * (x @ x.T)
    k_mat = np.exp(-d2 / (2.0 * 3.0))
    m = 60
    idx = rng.choice(n, m, replace=False)
    approx = nystrom(k_mat, idx)
    err = float(np.linalg.norm(k_mat - approx) / np.linalg.norm(k_mat))
    # eigengap check: spectrum captured
    ev_true = np.linalg.eigvalsh(k_mat)[-10:]
    ev_ny = np.linalg.eigvalsh(approx)[-10:]
    eig_err = float(np.max(np.abs(ev_true - ev_ny) / np.maximum(ev_true, 1e-9)))
    return {
        "synthetic_nystrom_err": err,
        "synthetic_eig_err": eig_err,
        "synthetic_landmarks": float(m),
    }
