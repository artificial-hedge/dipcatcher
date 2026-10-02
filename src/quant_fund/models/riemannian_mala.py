"""Riemannian (metric-adapted) MALA — position-dependent diagonal
metric M(x) = diag(1 + |∂²U|) preconditions proposals on the curved
banana: proposal x + h/2 M⁻¹∇logp + √h M⁻¹ᐟ²z. ESS vs RWM.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._pd_synth import (
    DIM,
    grad_logp,
    logp,
    mean_ess,
    moment_err,
    ref_moments,
    rwm_baseline,
)

FloatArray = NDArray[np.float64]


def _metric(x: FloatArray) -> FloatArray:
    g = grad_logp(x)
    # crude curvature proxy: per-coord outer-product diag + ridge
    return 1.0 + g * g + np.abs(x) * 0.5


def _prop(x: FloatArray, h: float, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    m = _metric(x)
    mean = x + 0.5 * h * grad_logp(x) / m
    z = rng.standard_normal(DIM) * np.sqrt(h / m)
    return mean + z, m


def _lq(y: FloatArray, x: FloatArray, h: float) -> float:
    m = _metric(x)
    mean = x + 0.5 * h * grad_logp(x) / m
    d = y - mean
    return float(-0.5 * np.sum(d * d * m / h))


def _rmala(seed: int, n: int = 12000, h: float = 0.25) -> FloatArray:
    rng = np.random.default_rng(seed)
    x = np.zeros(DIM)
    out = np.zeros((n, DIM))
    for i in range(n):
        y, _ = _prop(x, h, rng)
        a = logp(y) + _lq(x, y, h) - logp(x) - _lq(y, x, h)
        if np.log(rng.random()) < a:
            x = y
        out[i] = x
    return out


def bench_riemannian_mala(seed: int = 2233) -> dict[str, float]:
    mu, sd = ref_moments()
    smp = _rmala(seed)
    base = rwm_baseline(seed + 1, n=len(smp))
    ess = mean_ess(smp)
    ess_b = mean_ess(base)
    return {
        "synthetic_rmala_ess": ess,
        "synthetic_rwm_ess": ess_b,
        "synthetic_rmala_ess_gain": ess - ess_b,
        "synthetic_rmala_moment_err": moment_err(smp, mu, sd),
        "torch_available": 0.0,
    }
