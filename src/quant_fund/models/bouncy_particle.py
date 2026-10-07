"""Bouncy Particle Sampler (Bouchard-Côté et al. 2018) — non-reversible (SYNTHETIC)
PDMP: straight-line trajectories x+vt, bounce events at rate
max(0, v·∇U) via thinning, plus Poisson velocity refresh. ESS vs RWM.
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


def _bound(x: FloatArray) -> float:
    """Heuristic sup bound on |∇U| near x (banana curvature ~ x²/BAND²)."""
    return 4.0 * (np.sum(x * x) + 4.0) + 8.0


def _bps(seed: int, horizon: float = 400.0, refresh: float = 1.0) -> FloatArray:
    rng = np.random.default_rng(seed)
    x = np.zeros(DIM)
    v = rng.standard_normal(DIM)
    v /= np.linalg.norm(v)
    t = 0.0
    out = []
    while t < horizon:
        gu = -grad_logp(x)  # ∇U
        lam_max = float(abs(v @ gu) + np.linalg.norm(v) * _bound(x) * 0.25)
        tau = rng.exponential(1.0 / max(lam_max, 1e-6))
        if t + tau > horizon:
            break
        x = x + v * tau
        t += tau
        gu = -grad_logp(x)
        lam = max(0.0, float(v @ gu))
        if rng.random() < lam / lam_max:
            # bounce: reflect v against ∇U
            n = gu / (np.linalg.norm(gu) + 1e-12)
            v = v - 2.0 * (v @ n) * n
        if rng.random() < refresh * tau:
            v = rng.standard_normal(DIM)
            v /= np.linalg.norm(v)
        out.append(x.copy())
    return np.asarray(out)


def bench_bouncy_particle(seed: int = 2201) -> dict[str, float]:
    mu, sd = ref_moments()
    smp = _bps(seed)
    base = rwm_baseline(seed + 1, n=len(smp))
    ess = mean_ess(smp)
    ess_b = mean_ess(base)
    return {
        "synthetic_bps_ess": ess,
        "synthetic_rwm_ess": ess_b,
        "synthetic_bps_ess_gain": ess - ess_b,
        "synthetic_bps_moment_err": moment_err(smp, mu, sd),
        "synthetic_bps_accept_events": float(len(smp)),
        "synthetic_torch_available": 0.0,
    }
