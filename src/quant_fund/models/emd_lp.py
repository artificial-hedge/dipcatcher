"""Earthmover canon: 1-D EMD via the CDF closed form, (SYNTHETIC)
exact n-D EMD via the transport LP (HiGHS), and the
Bures-Wasserstein closed form between Gaussians — cross-
checked on a synthetic two-sample fixture.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.stats import wasserstein_distance

FloatArray = np.ndarray


def emd_1d(x: FloatArray, y: FloatArray) -> float:
    """1-D EMD (W1) via sorted CDF area — matches scipy."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    y = np.sort(np.asarray(y, dtype=np.float64))
    xs = np.sort(np.concatenate([x, y]))
    fx = np.searchsorted(x, xs[:-1], side="right") / x.size
    fy = np.searchsorted(y, xs[:-1], side="right") / y.size
    return float(np.sum(np.abs(fx - fy) * np.diff(xs)))


def emd_lp(xs: FloatArray, xt: FloatArray) -> float:
    """n-D EMD between equal-mass point clouds via transport LP."""
    xs = np.asarray(xs, dtype=np.float64)
    xt = np.asarray(xt, dtype=np.float64)
    if np.allclose(xs, xt):
        return 0.0
    m, n = xs.shape[0], xt.shape[0]
    cost = np.sqrt(
        np.maximum(
            np.sum(xs**2, axis=1, keepdims=True) + np.sum(xt**2, axis=1)[None, :] - 2.0 * xs @ xt.T,
            0.0,
        )
    )
    a_eq = np.zeros((m + n, m * n))
    for i in range(m):
        a_eq[i, i * n : (i + 1) * n] = 1.0
    for j in range(n):
        a_eq[m + j, j::n] = 1.0
    b_eq = np.concatenate([np.full(m, 1.0 / m), np.full(n, 1.0 / n)])
    res = linprog(
        cost.ravel(),
        A_eq=a_eq[:-1],
        b_eq=b_eq[:-1],
        bounds=(0.0, None),
        method="highs",
    )
    if res.status != 0:
        raise RuntimeError(f"emd LP failed: {res.message}")
    return float(res.fun)


def bures_wasserstein(m0: FloatArray, s0: FloatArray, m1: FloatArray, s1: FloatArray) -> float:
    """W2 between Gaussians N(m0,S0), N(m1,S1)."""
    m0 = np.asarray(m0, dtype=np.float64)
    m1 = np.asarray(m1, dtype=np.float64)
    s0 = np.asarray(s0, dtype=np.float64)
    s1 = np.asarray(s1, dtype=np.float64)
    s1h = np.linalg.cholesky(s1 + 1e-12 * np.eye(s1.shape[0]))
    mid = s1h @ s0 @ s1h.T
    ev = np.linalg.eigvalsh(mid)
    tr_term = float(np.trace(s0 + s1) - 2.0 * np.sum(np.sqrt(np.maximum(ev, 0.0))))
    w2 = float(m0 @ m0 - 2.0 * m0 @ m1 + m1 @ m1 + tr_term)
    return float(np.sqrt(max(w2, 0.0)))


def bench_emd_lp(seed: int | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else 20261231)
    n = 40
    x = rng.standard_normal(n)
    y = rng.standard_normal(n) + 0.7
    e1 = emd_1d(x, y)
    e_ref = float(wasserstein_distance(x, y))
    xs2 = rng.standard_normal((30, 2))
    xt2 = rng.standard_normal((30, 2)) + np.array([0.8, -0.4])
    e2 = emd_lp(xs2, xt2)
    m0 = np.zeros(3)
    s0 = np.eye(3)
    m1 = np.array([1.0, 0.0, 0.0])
    s1 = np.diag([4.0, 1.0, 0.25])
    bw = bures_wasserstein(m0, s0, m1, s1)
    # closed-form check: mean diff^2 + trace(1+4-2*sqrt(4)) + trace(1+1-2) + trace(1+0.25-1)
    bw_ref = np.sqrt(1.0 + (1 + 4 - 2 * 2.0) + (1 + 1 - 2 * 1.0) + (1 + 0.25 - 2 * 0.5))
    return {
        "synthetic_emd1d": e1,
        "synthetic_emd1d_err": abs(e1 - e_ref),
        "synthetic_emd_lp": e2,
        "synthetic_bures": bw,
        "synthetic_bures_err": abs(bw - float(bw_ref)),
    }
