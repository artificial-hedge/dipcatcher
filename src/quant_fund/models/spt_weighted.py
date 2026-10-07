"""WSPT (weighted shortest processing time, Smith 1956): optimal
single-machine Σw_jC_j sequencing by p_j/w_j order vs FIFO/random.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._sched_synth import weighted_jobs


def _wcj(order: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    c = 0.0
    tot = 0.0
    for j in order:
        c += p[j]
        tot += w[j] * c
    return tot


def bench_spt_weighted(seed: int = 3061) -> dict[str, float]:
    p, w = weighted_jobs(seed)
    wspt = np.argsort(p / w)
    fifo = np.arange(len(p))
    rng = np.random.default_rng(seed)
    rnd = rng.permutation(len(p))
    c_opt = _wcj(wspt, p, w)
    c_fifo = _wcj(fifo, p, w)
    c_rnd = _wcj(rnd, p, w)
    return {
        "synthetic_wspt_cost": float(c_opt),
        "synthetic_fifo_cost": float(c_fifo),
        "synthetic_random_cost": float(c_rnd),
        "synthetic_wspt_gain": float(c_fifo - c_opt),
        "synthetic_torch_available": 0.0,
    }
