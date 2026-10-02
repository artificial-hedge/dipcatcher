"""Preconditioned Crank–Nicolson (Cotter et al. 2013): dimension-robust
function-space proposal u' = sqrt(1-beta²)u + beta·xi on a GP target —
acceptance stays healthy where RWM dies as dimension grows.
"""

from __future__ import annotations

import numpy as np


def bench_pcn_sampler(
    seed: int = 2981, d: int = 60, steps: int = 800, beta: float = 0.08
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    # target: discretized GP on [0,1], SE kernel ell=0.2 + tiny obs of sin(2πx)
    xs = np.linspace(0, 1, d)
    K = np.exp(-((xs[:, None] - xs[None, :]) ** 2) / (2 * 0.2**2)) + 1e-8 * np.eye(d)
    L = np.linalg.cholesky(K)
    yobs = np.sin(2 * np.pi * xs[::10]) + 0.15 * rng.standard_normal(6)

    def logp(u: np.ndarray) -> float:
        z = np.linalg.solve(L, u)
        return float(-0.5 * z @ z - 0.5 / 0.15**2 * ((u[::10] - yobs) ** 2).sum())

    u = np.zeros(d)
    acc = 0
    samp = []
    for _ in range(steps):
        xi = L @ rng.standard_normal(d)
        up = np.sqrt(1 - beta**2) * u + beta * xi
        if np.log(rng.uniform()) < logp(up) - logp(u):
            u = up
            acc += 1
        samp.append(u.copy())
    samp_a = np.asarray(samp[200:])
    # posterior at obs points should track sin
    fit_err = float(np.sqrt(((samp_a[:, ::10].mean(0) - np.sin(2 * np.pi * xs[::10])) ** 2).mean()))
    return {
        "synthetic_pcn_accept": float(acc / steps),
        "synthetic_pcn_fit_err": fit_err,
        "synthetic_pcn_dim": float(d),
        "torch_available": 0.0,
    }
