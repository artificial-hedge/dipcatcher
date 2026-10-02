"""Wasserstein barycenter canon (Cuturi & Doucet 2014):
fixed-support entropic barycenter of histograms via
iterated kernel convolutions, on a synthetic dataset of
perturbed Gaussian mixtures on a 1-D grid.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def barycenter_fixed_support(
    measures: list[FloatArray],
    cost: FloatArray,
    eps: float = 0.02,
    weights: FloatArray | None = None,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> FloatArray:
    """Cuturi-Doucet fixed-support barycenter; returns weights."""
    ms = [np.asarray(m, dtype=np.float64) for m in measures]
    s = len(ms)
    n = ms[0].size
    w = np.asarray(weights, dtype=np.float64) if weights is not None else np.full(s, 1.0 / s)
    k = np.exp(-cost / eps)
    u = np.ones((s, n))
    v = np.ones((s, n))
    bar = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        bar_prev = bar.copy()
        for i in range(s):
            u[i] = ms[i] / np.maximum(k @ v[i], 1e-300)
        ktu = np.stack([k.T @ u[i] for i in range(s)])
        bar = np.exp(np.sum(w[:, None] * np.log(np.maximum(ktu, 1e-300)), axis=0))
        bar = bar / np.maximum(bar.sum(), 1e-300)
        for i in range(s):
            v[i] = bar / np.maximum(ktu[i], 1e-300)
        if np.max(np.abs(bar - bar_prev)) < tol:
            break
    return np.asarray(bar / bar.sum(), dtype=np.float64)


def bench_wasserstein_barycenter(seed: int | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else 20261231)
    n = 60
    grid = np.linspace(0.0, 1.0, n)
    cost = (grid[:, None] - grid[None, :]) ** 2

    def _mix(mu: float) -> FloatArray:
        d = np.exp(-0.5 * ((grid - mu) / 0.05) ** 2)
        d += 0.5 * np.exp(-0.5 * ((grid - mu - 0.15) / 0.07) ** 2)
        return np.asarray(d / d.sum(), dtype=np.float64)

    measures = [_mix(0.30 + 0.02 * float(rng.standard_normal())) for _ in range(3)]
    measures += [_mix(0.55 + 0.02 * float(rng.standard_normal())) for _ in range(3)]
    bar = barycenter_fixed_support(measures, cost, eps=0.004, max_iter=800)
    peak = float(grid[int(np.argmax(bar))])
    mass = float(bar.sum())
    # compare vs naive arithmetic mean
    am = np.mean(measures, axis=0)
    tv = float(0.5 * np.sum(np.abs(bar - am)))
    return {
        "synthetic_bary_mass": mass,
        "synthetic_bary_peak": peak,
        "synthetic_bary_tv_vs_mean": tv,
        "synthetic_bary_entropy": float(-np.sum(bar * np.log(np.maximum(bar, 1e-300)))),
    }
