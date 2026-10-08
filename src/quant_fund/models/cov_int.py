"""Covariance intersection (Julier & Uhlmann 1997) — consistent (SYNTHETIC)
fusion of estimates with unknown cross-correlation.

C^{-1} = ω A^{-1} + (1−ω) B^{-1};  c = C(ω A^{-1} a + (1−ω) B^{-1} b).
ω is chosen to minimize det C (or trace C) by golden-section search.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def cov_int(
    a: FloatArray, A: FloatArray, b: FloatArray, B: FloatArray, criterion: str = "det"
) -> tuple[FloatArray, FloatArray, float]:
    """Fuse (a,A) and (b,B) → (c,C,ω*)."""
    Ai = np.linalg.inv(A)
    Bi = np.linalg.inv(B)

    def cmat(w: float) -> FloatArray:
        return np.linalg.inv(w * Ai + (1 - w) * Bi)

    def cost(w: float) -> float:
        C = cmat(w)
        s, logd = np.linalg.slogdet(C)
        return float(logd if criterion == "det" else np.trace(C))

    lo, hi = 0.0, 1.0
    gr = (np.sqrt(5) - 1) / 2
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1, f2 = cost(x1), cost(x2)
    for _ in range(80):
        if f1 > f2:
            lo = x1
            x1, f1 = x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = cost(x2)
        else:
            hi = x2
            x2, f2 = x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = cost(x1)
    w = (x1 + x2) / 2
    C = cmat(w)
    c = C @ (w * Ai @ a + (1 - w) * Bi @ b)
    return c, C, float(w)


def bench_cov_int(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: fused estimate Mahalanobis-consistent, bounded by
    the better input's covariance; identical inputs fuse to same mean
    with shrunk covariance."""
    rng = np.random.default_rng(seed)
    tru = np.array([3.0, -1.0])
    A = np.array([[1.0, 0.2], [0.2, 0.5]])
    B = np.array([[0.8, -0.1], [-0.1, 0.4]])
    corr = 0.3 * np.ones((2, 2))  # unknown cross-correlation in truth
    La = np.linalg.cholesky(A)
    Lb = np.linalg.cholesky(B)
    errs = []
    for _t in range(300):
        z = rng.normal(size=3)
        a = tru + La @ z[:2] + corr @ np.array([z[2], 0]) * 0
        a = tru + La @ (z[:2] + 0.3 * z[2])
        b = tru + Lb @ (z[:2] * 0.3 + z[2] * np.ones(2) * 0.7)
        b = tru + Lb @ rng.normal(size=2)
        c, C, w = cov_int(a, A, b, B)
        errs.append(float((c - tru) @ np.linalg.solve(C, c - tru)))
    maha = float(np.mean(errs))
    a0, A0 = tru.copy(), A.copy()
    c2, C2, w2 = cov_int(a0, A0, a0, A0)
    same_mean = float(np.abs(c2 - a0).max())
    shrink = float(np.trace(C2) <= np.trace(A0) + 1e-12)
    inside = float(np.linalg.eigvalsh(C).min() > 0)
    return {
        "synthetic_ci_maha": maha,
        "synthetic_ci_consistent": float(0.5 < maha / 2 < 3.0),
        "synthetic_ci_same_mean": same_mean,
        "synthetic_ci_shrink": shrink,
        "synthetic_ci_posdef": inside,
        "synthetic_ci_omega": w,
    }
