"""LPT (longest-processing-time) list scheduling on identical parallel
machines — makespan vs lower bound max(Σp/m, p_max); ~4/3-OPT guarantee.
"""

from __future__ import annotations

import numpy as np


def _lpt(p: np.ndarray, m: int) -> float:
    loads = np.zeros(m)
    for j in np.argsort(-p):
        loads[np.argmin(loads)] += p[j]
    return float(loads.max())


def _list(p: np.ndarray, m: int, order: np.ndarray) -> float:
    loads = np.zeros(m)
    for j in order:
        loads[np.argmin(loads)] += p[j]
    return float(loads.max())


def bench_lpt_schedule(seed: int = 3049, n: int = 60, m: int = 4) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    p = rng.uniform(1, 20, n)
    lb = max(p.sum() / m, p.max())
    ms_lpt = _lpt(p, m)
    ms_rnd = _list(p, m, rng.permutation(n))
    return {
        "synthetic_lpt_ms": ms_lpt,
        "synthetic_random_ms": float(ms_rnd),
        "synthetic_lpt_lb_ratio": float(ms_lpt / lb),
        "synthetic_lpt_gain": float(ms_rnd - ms_lpt),
        "torch_available": 0.0,
    }
