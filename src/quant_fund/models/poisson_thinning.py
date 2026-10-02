"""Lewis–Shedler thinning for an inhomogeneous Poisson process:
rate λ(t) = a + b·sin(2πt). Thinned event times should match the
theoretical intensity profile; flat-homogeneous sampler baseline.
"""

from __future__ import annotations

import numpy as np


def _rate(t: np.ndarray) -> np.ndarray:
    return 5.0 * (1.0 + 0.8 * np.sin(2 * np.pi * t))


def bench_poisson_thinning(seed: int = 2943, T: float = 20.0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    # thinning
    events: list[float] = []
    t = 0.0
    lam_max = 9.0
    while t < T:
        t += rng.exponential(1 / lam_max)
        if t >= T:
            break
        if rng.uniform() < _rate(np.asarray([t]))[0] / lam_max:
            events.append(t)
    ev = np.asarray(events)
    # expected count = integral of rate = 4.0
    exp_n = 5.0 * T
    # event density correlation with rate profile (bin centers)
    bins = np.linspace(0, T, 81)
    hist = np.histogram(ev, bins)[0] * (80 / T)
    rate_at = _rate((bins[:-1] + bins[1:]) / 2)
    corr = float(np.corrcoef(hist, rate_at)[0, 1])
    # homogeneous baseline sampled uniformly
    ev_b = rng.uniform(0, T, len(ev))
    hist_b = np.histogram(ev_b, bins)[0] * (80 / T)
    corr_b = float(np.corrcoef(hist_b, rate_at)[0, 1])
    return {
        "synthetic_thin_count_err": float(abs(len(ev) - exp_n) / exp_n),
        "synthetic_thin_rate_corr": corr,
        "synthetic_unif_rate_corr": corr_b,
        "torch_available": 0.0,
    }
