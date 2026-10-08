"""Ogata thinning for an exponential Hawkes process: (SYNTHETIC)
λ(t) = μ + Σ_{t_i<t} α e^{-β(t-t_i)}. Clustering signature: event-count
bursts vs homogeneous Poisson at same mean; branching ratio α/β < 1.
"""

from __future__ import annotations

import numpy as np


def _hawkes(mu: float, alpha: float, beta: float, T: float, rng) -> np.ndarray:
    t = 0.0
    events: list[float] = []
    while t < T:
        lam = mu + sum(alpha * np.exp(-beta * (t - e)) for e in events)
        lam_next = max(lam, mu)
        t += rng.exponential(1 / lam_next)
        if t >= T:
            break
        lam_t = mu + sum(alpha * np.exp(-beta * (t - e)) for e in events)
        if rng.uniform() * lam_next <= lam_t:
            events.append(t)
    return np.asarray(events)


def bench_hawkes_thinning(seed: int = 2947, T: float = 100.0, trials: int = 30) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    mu, alpha, beta = 1.0, 0.6, 1.0
    counts, burt = [], []
    for _ in range(trials):
        ev = _hawkes(mu, alpha, beta, T, rng)
        counts.append(len(ev))
        # burstiness B = (sd-mean)/(sd+mean) of inter-event times
        ie = np.diff(ev)
        if len(ie) > 2:
            burt.append((ie.std() - ie.mean()) / (ie.std() + ie.mean() + 1e-9))
    mean_n = np.mean(counts)
    # homogeneous Poisson at same rate for comparison
    hp = rng.poisson(mean_n / T * 10, 10000)
    var_hom = hp.var() / hp.mean()  # ~1
    var_haw = np.var(counts) / (mean_n + 1e-9)
    return {
        "synthetic_hawkes_fano": float(var_haw),
        "synthetic_poisson_fano": float(var_hom),
        "synthetic_hawkes_burstiness": float(np.mean(burt)),
        "synthetic_hawkes_mean_rate_err": float(
            abs(mean_n / T - mu / (1 - alpha / beta)) / (mu / (1 - alpha / beta))
        ),
        "synthetic_torch_available": 0.0,
    }
