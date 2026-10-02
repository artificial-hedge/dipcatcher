"""Shared fixture for wave-180 PDMP / exotic-sampling canon.

Banana posterior in d=4 (Rosenbrock-style): p(x) ∝ exp(-0.5*x0² -0.5*Σ (x_i - (x_{i-1}² + 1))²/0.1²).
Analytic gradient available; target moment ground truth from a long
overdamped reference chain. Baseline: random-walk Metropolis.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

DIM = 4
BAND = 0.35  # curvature bandwidth


def logp(x: FloatArray) -> float:
    out = -0.5 * x[0] ** 2
    for i in range(1, len(x)):
        out -= 0.5 * (x[i] - (x[i - 1] ** 2 + 1.0)) ** 2 / BAND**2
    return float(out)


def grad_logp(x: FloatArray) -> FloatArray:
    g = np.zeros(len(x))
    g[0] = -x[0]
    for i in range(1, len(x)):
        r = x[i] - (x[i - 1] ** 2 + 1.0)
        g[i] -= r / BAND**2
        g[i - 1] += 2.0 * x[i - 1] * r / BAND**2
    return g


def ref_moments(seed: int = 999, n: int = 40000) -> tuple[FloatArray, FloatArray]:
    """Reference mean/std from a long fine-step MALA chain."""
    rng = np.random.default_rng(seed)
    x = np.zeros(DIM)
    h = 0.03
    acc = np.zeros((n, DIM))
    for i in range(n):
        z = rng.standard_normal(DIM)
        gx = grad_logp(x)
        y = x + 0.5 * h * gx + np.sqrt(h) * z
        gy = grad_logp(y)
        lq_yx = -0.5 * np.sum((y - x - 0.5 * h * gx) ** 2) / h
        lq_xy = -0.5 * np.sum((x - y - 0.5 * h * gy) ** 2) / h
        if np.log(rng.random()) < logp(y) + lq_xy - logp(x) - lq_yx:
            x = y
        acc[i] = x
    acc = acc[n // 4 :: 2]
    return acc.mean(0), acc.std(0)


def rwm_baseline(seed: int, n: int = 12000, step: float = 0.35) -> FloatArray:
    rng = np.random.default_rng(seed)
    x = np.zeros(DIM)
    out = np.zeros((n, DIM))
    for i in range(n):
        y = x + rng.standard_normal(DIM) * step
        if np.log(rng.random()) < logp(y) - logp(x):
            x = y
        out[i] = x
    return out


def moment_err(samples: FloatArray, mu: FloatArray, sd: FloatArray) -> float:
    m = samples[len(samples) // 4 :]
    return float(np.abs(m.mean(0) - mu).mean() + np.abs(m.std(0) - sd).mean())


def ess_1d(x: FloatArray) -> float:
    """IACT-based effective sample size (autocorr until negative)."""
    n = len(x)
    if n < 20:
        return float(n)
    xc = x - x.mean()
    v = float(xc @ xc / n)
    if v <= 0:
        return float(n)
    s = 1.0
    for k in range(1, min(n - 1, 200)):
        r = float(xc[:-k] @ xc[k:] / (n - k) / float(v))
        if r < 0:
            break
        s += 2 * r
    return float(min(n, n / max(s, 1.0)))


def mean_ess(samples: FloatArray) -> float:
    m = samples[len(samples) // 4 :]
    return float(np.mean([ess_1d(m[:, j]) for j in range(m.shape[1])]))
