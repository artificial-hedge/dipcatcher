"""Boomerang Sampler (Bierkens et al. 2020) — PDMP on elliptical
(rotation) dynamics: x(t)=x0 cos t + v0 sin t, bounce rate
max(0, <v, ∇U>) via thinning. ESS vs RWM.
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


def _boomerang(seed: int, horizon: float = 200.0) -> FloatArray:
    rng = np.random.default_rng(seed)
    x = np.zeros(DIM)
    v = rng.standard_normal(DIM)
    t = 0.0
    out = []
    while t < horizon:
        gu = -grad_logp(x)
        lam_max = float(
            np.linalg.norm(v)
            * (np.linalg.norm(gu) + 4.0 * (np.sum(x * x) + np.sum(v * v) + 4.0))
            * 0.5
            + 1.0
        )
        tau = rng.exponential(1.0 / lam_max)
        if t + tau > horizon:
            break
        x_new = x * np.cos(tau) + v * np.sin(tau)
        v_new = -x * np.sin(tau) + v * np.cos(tau)
        t += tau
        gu = -grad_logp(x_new)
        lam = max(0.0, float(v_new @ gu))
        if rng.random() < lam / lam_max:
            n = gu / (np.linalg.norm(gu) + 1e-12)
            v_new = v_new - 2.0 * (v_new @ n) * n
        x, v = x_new, v_new
        out.append(x.copy())
    return np.asarray(out)


def bench_boomerang_sampler(seed: int = 2213) -> dict[str, float]:
    mu, sd = ref_moments()
    smp = _boomerang(seed)
    base = rwm_baseline(seed + 1, n=len(smp))
    ess = mean_ess(smp)
    ess_b = mean_ess(base)
    return {
        "synthetic_boom_ess": ess,
        "synthetic_rwm_ess": ess_b,
        "synthetic_boom_ess_gain": ess - ess_b,
        "synthetic_boom_moment_err": moment_err(smp, mu, sd),
        "synthetic_torch_available": 0.0,
    }
