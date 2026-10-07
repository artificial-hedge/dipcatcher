"""Feynman-Kac Monte Carlo — u(t,x) = E[u0(x + σW_t)] estimated by
Brownian sampling at each grid point; stochastic baseline for the PDE.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._heat_synth import SIG, T_, eval_error, grid, u0


def bench_feynman_kac_mc(seed: int = 2531, n_mc: int = 4000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x = grid()
    uT = np.zeros_like(x)
    for i, xi in enumerate(x):
        w = xi + SIG * np.sqrt(T_) * rng.standard_normal(n_mc)
        uT[i] = float(u0(w).mean())

    def pred(xq: np.ndarray, tq: float) -> np.ndarray:
        return np.asarray(np.interp(xq, x, uT))

    return {"synthetic_fkmc_rel_l2": eval_error(pred), "synthetic_torch_available": 0.0}
