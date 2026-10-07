"""MPPI (Williams et al. 2017): sampled-trajectory control via
exponentially-weighted noise rollouts on the double integrator.
Cost vs PD baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._oc_synth import HORIZON, dyn, pd_baseline, stage_cost


def bench_mppi_control(
    seed: int = 2917, n_samples: int = 64, lam: float = 1.0, sigma: float = 0.8
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x0 = np.array([0.0, 0.0])
    U = np.zeros(HORIZON)
    for _ in range(3):  # refine U via MPPI iterations
        eps = rng.normal(0, sigma, (n_samples, HORIZON))
        costs = np.zeros(n_samples)
        for k in range(n_samples):
            x = x0.copy()
            for t in range(HORIZON):
                u = U[t] + eps[k, t]
                x = dyn(x, u)
                costs[k] += stage_cost(x, u)
        w = np.exp(-(costs - costs.min()) / lam)
        w /= w.sum()
        U = (w[:, None] * (U[None, :] + eps)).sum(0)
    # final cost under mean plan
    x = x0.copy()
    tot = 0.0
    for t in range(HORIZON):
        x = dyn(x, U[t])
        tot += stage_cost(x, U[t])
    _, cost_b = pd_baseline(x0)
    return {
        "synthetic_mppi_cost": float(tot),
        "synthetic_pd_cost": float(cost_b),
        "synthetic_mppi_gain": float(cost_b - tot),
        "synthetic_torch_available": 0.0,
    }
