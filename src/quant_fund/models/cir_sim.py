"""CIR process exact/noncentral-χ² simulation vs Euler — verifies the (SYNTHETIC)
exact scheme's mean/variance against analytic CIR moments where Euler
drifts and goes negative.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ncx2

FloatArray = NDArray[np.float64]


def _cir_exact(r0: float, k: float, th: float, xi: float, T: float, n: int, rng) -> FloatArray:
    d = 4 * k * th / xi**2
    c = xi**2 * (1 - np.exp(-k * T)) / (4 * k)
    lam = 4 * k * np.exp(-k * T) * r0 / (xi**2 * (1 - np.exp(-k * T)))
    return np.asarray(c * ncx2.rvs(d, lam, size=n, random_state=rng))


def _cir_euler(
    r0: float, k: float, th: float, xi: float, T: float, n: int, steps: int, rng
) -> FloatArray:
    dt = T / steps
    r = np.full(n, r0)
    for _ in range(steps):
        r = (
            r
            + k * (th - r) * dt
            + xi * np.sqrt(np.maximum(r, 0)) * np.sqrt(dt) * rng.standard_normal(n)
        )
    return np.asarray(r)


def bench_cir_sim(seed: int = 2933, n: int = 4000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    k, th, xi, r0, T = 0.8, 0.05, 0.25, 0.03, 1.0
    exact = _cir_exact(r0, k, th, xi, T, n, rng)
    euler = _cir_euler(r0, k, th, xi, T, n, 50, rng)
    mean_t = th + (r0 - th) * np.exp(-k * T)
    var_t = (
        r0 * xi**2 / k * (np.exp(-k * T) - np.exp(-2 * k * T))
        + th * xi**2 / (2 * k) * (1 - np.exp(-k * T)) ** 2
    )
    return {
        "synthetic_cir_exact_mean_err": float(abs(exact.mean() - mean_t)),
        "synthetic_cir_euler_mean_err": float(abs(euler.mean() - mean_t)),
        "synthetic_cir_exact_var_err": float(abs(exact.var() - var_t)),
        "synthetic_cir_euler_neg_frac": float((euler < 0).mean()),
        "synthetic_torch_available": 0.0,
    }
