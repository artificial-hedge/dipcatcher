"""Zig-Zag Sampler (Bierkens et al. 2019) — coordinate-wise PDMP:
v_i ∈ {±1}, event i at rate max(0, v_i ∂_i U) flips v_i. ESS vs RWM.
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


def _zigzag(seed: int, horizon: float = 800.0) -> FloatArray:
    rng = np.random.default_rng(seed)
    x = np.zeros(DIM)
    v = rng.choice([-1.0, 1.0], DIM)
    t = 0.0
    out = []
    while t < horizon:
        gu = -grad_logp(x)
        rates = np.maximum(0.0, v * gu)
        # thinning bound per coordinate: |∂iU| ≤ bound(x)+|gi|
        bound = np.abs(gu) + 4.0 * (np.sum(x * x) + 4.0)
        lam = np.maximum(rates, 1e-9) + bound * 0.25
        taus = rng.exponential(1.0 / np.maximum(lam, 1e-9))
        i = int(np.argmin(taus))
        tau = float(taus[i])
        if t + tau > horizon:
            break
        x = x + v * tau
        t += tau
        lam_true = max(0.0, float(v[i] * (-grad_logp(x)[i])))
        if rng.random() < lam_true / lam[i]:
            v[i] = -v[i]
        out.append(x.copy())
    return np.asarray(out)


def bench_zigzag_sampler(seed: int = 2207) -> dict[str, float]:
    mu, sd = ref_moments()
    smp = _zigzag(seed)
    base = rwm_baseline(seed + 1, n=len(smp))
    ess = mean_ess(smp)
    ess_b = mean_ess(base)
    return {
        "synthetic_zz_ess": ess,
        "synthetic_rwm_ess": ess_b,
        "synthetic_zz_ess_gain": ess - ess_b,
        "synthetic_zz_moment_err": moment_err(smp, mu, sd),
        "torch_available": 0.0,
    }
