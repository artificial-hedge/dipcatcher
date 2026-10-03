"""CUR matrix-decomposition canon (Mahoney & Drineas 2009):
A ≈ C U R where C/R are actual columns/rows selected by
leverage-score sampling, vs truncated-SVD optimal error on
a synthetic low-rank+noise matrix.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def leverage_scores(a: FloatArray, k: int) -> FloatArray:
    """Column leverage scores from top-k right singular vectors."""
    _, _, vt = np.linalg.svd(a, full_matrices=False)
    lev = np.sum(vt[:k] ** 2, axis=0)
    return np.asarray(lev / lev.sum(), dtype=np.float64)


def cur_decomp(
    a: FloatArray,
    k: int,
    rng: np.random.Generator,
    n_cols: int | None = None,
    n_rows: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """CUR decomposition with leverage-score column selection."""
    a = np.asarray(a, dtype=np.float64)
    m, n = a.shape
    nc = n_cols if n_cols is not None else 3 * k
    nr = n_rows if n_rows is not None else 3 * k
    col_p = leverage_scores(a, k)
    cols = rng.choice(n, size=min(nc, n), replace=False, p=col_p)
    c = a[:, cols]
    row_lev = np.sum(np.linalg.svd(a.T, full_matrices=False)[2][:k] ** 2, axis=0)
    row_p = row_lev / row_lev.sum()
    rows = rng.choice(m, size=min(nr, m), replace=False, p=row_p)
    r = a[rows, :]
    u = np.linalg.pinv(c) @ a @ np.linalg.pinv(r)
    return c, u, r


def bench_cur_decomp(seed: int | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else 20261231)
    m, n, r = 300, 250, 12
    u0, _ = np.linalg.qr(rng.standard_normal((m, r)))
    v0, _ = np.linalg.qr(rng.standard_normal((n, r)))
    a = u0 @ np.diag(np.geomspace(20.0, 0.5, r)) @ v0.T
    a += 0.08 * rng.standard_normal((m, n))
    k = 12
    c, u, rr = cur_decomp(a, k, rng, n_cols=48, n_rows=48)
    err = float(np.linalg.norm(a - c @ u @ rr))
    u_ex, s_ex, vt_ex = np.linalg.svd(a, full_matrices=False)
    opt = u_ex[:, :k] @ np.diag(s_ex[:k]) @ vt_ex[:k]
    opt_err = float(np.linalg.norm(a - opt))
    return {
        "synthetic_cur_err": err,
        "synthetic_opt_err": opt_err,
        "synthetic_ratio": err / max(opt_err, 1e-12),
    }
