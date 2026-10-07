"""Zig-Zag Sampler (Bierkens et al. 2019) — coordinate-wise PDMP: (SYNTHETIC)
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


def _zigzag_trace(seed: int, horizon: float = 800.0) -> tuple[FloatArray, FloatArray]:
    """Run the skeleton; return (times, positions) at every event. Positions are
    piecewise linear between events along the active velocity."""
    rng = np.random.default_rng(seed)
    x = np.zeros(DIM)
    v = rng.choice([-1.0, 1.0], DIM)
    t = 0.0
    times = [0.0]
    pos = [x.copy()]
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
        times.append(t)
        pos.append(x.copy())
    return np.asarray(times), np.asarray(pos)


def _zigzag(seed: int, horizon: float = 800.0, n_grid: int = 4000) -> FloatArray:
    """Uniform-time resample of the skeleton. Event-time subsamples are biased
    for PDMPs (events are not a Poisson time process); moments/ESS must be
    computed on time-uniform positions, reconstructed exactly since the path
    is linear between events."""
    times, pos = _zigzag_trace(seed, horizon)
    if times.size < 2:
        return pos
    grid = np.linspace(0.0, times[-1], min(n_grid, times.size * 4))
    idx = np.searchsorted(times, grid, side="right") - 1
    idx = np.clip(idx, 0, times.size - 2)
    # velocity on segment k = (pos[k+1]-pos[k]) / (times[k+1]-times[k])
    seg_dt = times[idx + 1] - times[idx]
    frac = (grid - times[idx]) / np.maximum(seg_dt, 1e-12)
    return pos[idx] + (pos[idx + 1] - pos[idx]) * frac[:, None]


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
        "synthetic_torch_available": 0.0,
    }
