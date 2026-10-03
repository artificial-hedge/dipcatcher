"""Toeplitz canon: Levinson–Trench O(n²) recursion for Tx = b on
a strongly nonsingular Toeplitz matrix (Levinson–Durbin falls
out when b = −r[1:]); plus the Yule–Walker AR solve as the
symmetric special case. Bench: residual vs dense solve, AR
coefficient recovery vs Yule–Walker truth. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def levinson(r: FloatArray, b: FloatArray) -> FloatArray:
    """Levinson recursion for symmetric Toeplitz T(r) x = b,
    r = first row (r[0] > 0). O(n²).

    Maintains the forward-predictor a with T a = E·e_0; the
    particular solution updates through the reversed predictor.
    """
    r = np.asarray(r, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = b.size
    a = np.array([1.0])
    E = r[0]
    x = np.array([b[0] / r[0]])
    for k in range(1, n):
        # reflection coefficient for the Yule–Walker predictor
        delta = r[k] + float(np.dot(a[1:], r[k - 1 :: -1][: len(a) - 1]))
        kappa = -delta / E
        a = np.concatenate((a, [0.0])) + kappa * np.concatenate(([0.0], a[::-1]))
        E = E * (1.0 - kappa * kappa)
        # RHS drive via the reversed predictor
        lam = (b[k] - float(np.dot(x, r[k:0:-1]))) / E
        x = np.concatenate((x, [0.0])) + lam * a[::-1]
    return np.asarray(x, dtype=np.float64)


def yule_walker(r: FloatArray, p: int) -> FloatArray:
    """AR coefficients from autocovariances r[0..p] via Levinson
    on b = −r[1..p]."""
    return np.asarray(-levinson(r[: p + 1], -r[1 : p + 1]), dtype=np.float64)


def toeplitz_mat(r: FloatArray) -> FloatArray:
    """Dense symmetric Toeplitz from its first row."""
    n = r.size
    T = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            T[i, j] = r[abs(i - j)]
    return np.asarray(T, dtype=np.float64)


def bench_toeplitz_solve(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    # Toeplitz from an AR(1): r[k] = ρ^k — PD, well-conditioned
    n = 12
    rho = 0.6
    r = np.asarray(rho ** np.arange(n), dtype=np.float64)
    T = toeplitz_mat(r)
    b = rng.standard_normal(n)
    x = levinson(r, b)
    xd = np.linalg.solve(T, b)
    out["synthetic_levinson_err"] = float(np.linalg.norm(x - xd) / np.linalg.norm(xd))
    out["synthetic_levinson_residual"] = float(np.linalg.norm(T @ x - b) / np.linalg.norm(b))
    # Yule–Walker: AR(1) with true φ = ρ recovers [ρ, 0, 0, ...]
    a = yule_walker(r, 5)
    out["synthetic_yw_ar1_err"] = float(abs(a[0] - rho))
    out["synthetic_yw_tail_max"] = float(np.abs(a[1:]).max())
    # AR(2) truth: r from a known AR(2) autocov structure
    # use simulated AR(2) acf via Yule–Walker inverse
    phi1, phi2 = 0.5, 0.3
    # build acf by solving YW on a finite autocorr — use the
    # spectral approach: acf from statsmodels-free recursion
    c = np.zeros(8)
    c[0] = 1.0
    for k in range(1, 8):
        c[k] = phi1 * c[k - 1] + (phi2 * c[k - 2] if k >= 2 else 0.0)
    a2 = yule_walker(np.asarray(np.abs(c), dtype=np.float64), 3)
    out["synthetic_yw_ar2_1"] = float(a2[0])
    out["synthetic_yw_ar2_2"] = float(a2[1])
    return out
