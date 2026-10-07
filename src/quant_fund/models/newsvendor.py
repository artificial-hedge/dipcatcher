"""Newsvendor (Arrow et al. 1951): critical fractile Q* = F^{-1}(Cu/(Cu+Co)) (SYNTHETIC)
on Poisson demand vs naive mean-order.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson


def bench_newsvendor(seed: int = 3021, trials: int = 2000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    lam, cu, co = 20.0, 3.0, 1.0
    q_star = float(poisson.ppf(cu / (cu + co), lam))
    d = rng.poisson(lam, trials)

    def cost(q: float) -> float:
        return float((cu * np.maximum(d - q, 0) + co * np.maximum(q - d, 0)).mean())

    c_star = cost(q_star)
    c_mean = cost(lam)
    qs = np.arange(5, 45)
    cs = [cost(float(q)) for q in qs]
    q_emp = float(qs[int(np.argmin(cs))])
    return {
        "synthetic_nv_qstar": q_star,
        "synthetic_nv_q_emp": q_emp,
        "synthetic_nv_cost_star": c_star,
        "synthetic_nv_cost_naive": c_mean,
        "synthetic_nv_gain": float(c_mean - c_star),
        "synthetic_torch_available": 0.0,
    }
