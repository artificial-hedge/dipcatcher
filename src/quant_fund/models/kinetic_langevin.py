"""Kinetic (underdamped) Langevin Monte Carlo — velocity OU + position (SYNTHETIC)
drift discretization (randomized-horizon KLMC/UBU): partial momentum
refresh ρ per step. ESS vs RWM.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._pd_synth import (
    DIM,
    grad_logp,
    mean_ess,
    moment_err,
    ref_moments,
    rwm_baseline,
)

FloatArray = NDArray[np.float64]


def _klmc(seed: int, n: int = 12000, h: float = 0.05, gamma: float = 4.0) -> FloatArray:
    rng = np.random.default_rng(seed)
    x = np.zeros(DIM)
    v = rng.standard_normal(DIM)
    rho = np.exp(-gamma * h)
    out = np.zeros((n, DIM))
    for i in range(n):
        # UBU / OBABO-lite: half OU, drift, half OU
        v = rho * v + np.sqrt(1 - rho**2) * rng.standard_normal(DIM)
        x = x + h * v + 0.5 * h * h * grad_logp(x)
        v = rho * v + np.sqrt(1 - rho**2) * rng.standard_normal(DIM)
        out[i] = x
    return out


def bench_kinetic_langevin(seed: int = 2219) -> dict[str, float]:
    mu, sd = ref_moments()
    smp = _klmc(seed)
    base = rwm_baseline(seed + 1, n=len(smp))
    ess = mean_ess(smp)
    ess_b = mean_ess(base)
    return {
        "synthetic_klmc_ess": ess,
        "synthetic_rwm_ess": ess_b,
        "synthetic_klmc_ess_gain": ess - ess_b,
        "synthetic_klmc_moment_err": moment_err(smp, mu, sd),
        "synthetic_torch_available": 0.0,
    }
