"""Randomized SVD canon (Halko, Martinsson & Tropp 2011):
Gaussian range sketching with power iterations to capture
the dominant singular subspace, compared against exact
truncated SVD error on a synthetic low-rank+noise matrix.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def randomized_svd(
    a: FloatArray,
    rank: int,
    rng: np.random.Generator,
    oversample: int = 10,
    power: int = 2,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """HMT randomized SVD: A ≈ U diag(s) Vt."""
    a = np.asarray(a, dtype=np.float64)
    ell = rank + oversample
    omega = rng.standard_normal((a.shape[1], ell))
    y = a @ omega
    for _ in range(power):
        y = a @ (a.T @ y)
    q, _ = np.linalg.qr(y)
    b = q.T @ a
    ub, s, vt = np.linalg.svd(b, full_matrices=False)
    u = q @ ub
    return u[:, :rank], s[:rank], vt[:rank]


def bench_randomized_svd(seed: int | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else 20261231)
    n, p, r = 400, 300, 15
    u0, _ = np.linalg.qr(rng.standard_normal((n, r)))
    v0, _ = np.linalg.qr(rng.standard_normal((p, r)))
    sv = np.geomspace(50.0, 0.8, r)
    a = u0 @ np.diag(sv) @ v0.T + 0.05 * rng.standard_normal((n, p))
    k = 10
    u, s, vt = randomized_svd(a, k, rng, power=2)
    approx = u @ np.diag(s) @ vt
    u_ex, s_ex, vt_ex = np.linalg.svd(a, full_matrices=False)
    opt_err = float(np.linalg.norm(a - u_ex[:, :k] @ np.diag(s_ex[:k]) @ vt_ex[:k]))
    err = float(np.linalg.norm(a - approx))
    sv_err = float(np.max(np.abs(s - s_ex[:k]) / np.maximum(s_ex[:k], 1e-9)))
    return {
        "synthetic_rsvd_err": err,
        "synthetic_opt_err": opt_err,
        "synthetic_gap": err - opt_err,
        "synthetic_sv_err": sv_err,
    }
