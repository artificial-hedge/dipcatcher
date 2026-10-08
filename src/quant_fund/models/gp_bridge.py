"""GP bridge / conditional GP path: posterior conditioned on anchor (SYNTHETIC)
observations; uncertainty collapses at anchors and balloons mid-gap —
vs unconditional GP draw.
"""

from __future__ import annotations

import numpy as np


def _k(x: np.ndarray, y: np.ndarray, ell: float = 0.3) -> np.ndarray:
    return np.asarray(np.exp(-((x[:, None] - y[None, :]) ** 2) / (2 * ell**2)))


def bench_gp_bridge(seed: int = 2955, n_pts: int = 40) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 1, n_pts)
    # anchors at both ends + middle gap [0.4,0.6] unobserved
    xa = np.array([0.0, 0.25, 0.75, 1.0])
    ya = np.array([1.0, 0.5, -0.5, 0.0])
    Kaa = _k(xa, xa) + 1e-8 * np.eye(4)
    Ksa = _k(xs, xa)
    mu = Ksa @ np.linalg.solve(Kaa, ya)
    cov = _k(xs, xs) - Ksa @ np.linalg.solve(Kaa, Ksa.T)
    var = np.clip(np.diag(cov), 0, None)
    # anchor variance ~0
    anchor_var = float(np.mean([var[int(round(a * (n_pts - 1)))] for a in xa]))
    gap_var = float(var[int(0.5 * n_pts)])
    # unconditional draw variance (prior var = 1)
    uncond = float(np.ones(n_pts).mean())
    # sample paths
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(n_pts))
    samp = mu + L @ rng.standard_normal(n_pts)
    end_err = float(abs(samp[0] - 1.0) + abs(samp[-1] - 0.0))
    return {
        "synthetic_gp_anchor_var": anchor_var,
        "synthetic_gp_gap_var": gap_var,
        "synthetic_gp_uncond_var": uncond,
        "synthetic_gp_bridge_end_err": end_err,
        "synthetic_torch_available": 0.0,
    }
