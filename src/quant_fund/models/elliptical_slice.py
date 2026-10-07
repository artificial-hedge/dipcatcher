"""Elliptical Slice Sampling (Murray et al. 2010) — likelihood×Gaussian
prior posterior sampling via ellipse bracketing; no gradient needed.
ESS + moment error vs RWM on the banana likelihood.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._pd_synth import BAND, DIM, mean_ess, moment_err, ref_moments, rwm_baseline

FloatArray = NDArray[np.float64]


def _loglik(x: FloatArray) -> float:
    out = 0.0
    for i in range(1, len(x)):
        out -= 0.5 * (x[i] - (x[i - 1] ** 2 + 1.0)) ** 2 / BAND**2
    return float(out)


def _ess(seed: int, n: int = 8000) -> FloatArray:
    rng = np.random.default_rng(seed)
    x = np.zeros(DIM)
    out = np.zeros((n, DIM))
    for i in range(n):
        nu = rng.standard_normal(DIM)
        log_y = _loglik(x) + np.log(rng.random() + 1e-12)
        th = rng.uniform(0, 2 * np.pi)
        lo, hi = th - 2 * np.pi, th
        for _ in range(60):
            prop = x * np.cos(th) + nu * np.sin(th)
            if _loglik(prop) > log_y:
                x = prop
                break
            if th < 0:
                lo = th
            else:
                hi = th
            th = rng.uniform(lo, hi)
        out[i] = x
    return out


def bench_elliptical_slice(seed: int = 2227) -> dict[str, float]:
    mu, sd = ref_moments()
    smp = _ess(seed)
    base = rwm_baseline(seed + 1, n=len(smp))
    ess = mean_ess(smp)
    ess_b = mean_ess(base)
    return {
        "synthetic_ess_ess": ess,
        "synthetic_rwm_ess": ess_b,
        "synthetic_ess_ess_gain": ess - ess_b,
        "synthetic_ess_moment_err": moment_err(smp, mu, sd),
        "synthetic_torch_available": 0.0,
    }
