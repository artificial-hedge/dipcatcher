"""Reliability block diagram: series/parallel/k-out-of-n/cold standby (SYNTHETIC).

Exponential unit reliabilities R_i(t) = exp(-lam_i t). Series = prod;
parallel = 1-prod(1-R); k-of-n via the binomial tail; cold standby
(perfect switching, dormant units don't age) = Poisson-tailed survival
exp(-lam t) * sum_{j<k} (lam t)^j / j!. Each checked against Monte-Carlo
simulation on the shared fixture.
"""

import math

import numpy as np

from quant_fund.models._rel_synth import RBD_LAM, RBD_T


def _unit_r(lam: np.ndarray, t: float) -> np.ndarray:
    return np.exp(-lam * t)


def _k_out_of_n(r: np.ndarray, k: int) -> float:
    n = len(r)
    total = 0.0
    for mask in range(1 << n):
        up = bin(mask).count("1")
        pr = 1.0
        for i in range(n):
            pr *= r[i] if mask & (1 << i) else (1 - r[i])
        if up >= k:
            total += pr
    return total


def _standby(lam: float, t: float, n: int = 2) -> float:
    x = lam * t
    return float(np.exp(-x) * sum(x**j / math.factorial(j) for j in range(n)))


def _mc(lam: np.ndarray, t: float, n: int = 40000, seed: int = 2) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    life = rng.exponential(1.0 / lam[:, None], size=(len(lam), n))
    return {
        "series": float(np.mean(np.all(life > t, axis=0))),
        "parallel": float(np.mean(np.any(life > t, axis=0))),
        "k2": float(np.mean(np.sum(life > t, axis=0) >= 2)),
        "standby": float(np.mean(life[0] + rng.exponential(1.0 / lam[0], size=n) > t)),
    }


def bench_redundancy_block(seed: int = 4911) -> dict[str, float]:
    r = _unit_r(RBD_LAM, RBD_T)
    series = float(np.prod(r))
    parallel = float(1.0 - np.prod(1.0 - r))
    k2 = _k_out_of_n(r, 2)
    standby = _standby(float(RBD_LAM[0]), RBD_T)
    mc = _mc(RBD_LAM, RBD_T, seed=seed)
    return {
        "synthetic_rbd_series": series,
        "synthetic_rbd_series_mc_err": abs(series - mc["series"]),
        "synthetic_rbd_parallel": parallel,
        "synthetic_rbd_parallel_mc_err": abs(parallel - mc["parallel"]),
        "synthetic_rbd_k2": k2,
        "synthetic_rbd_k2_mc_err": abs(k2 - mc["k2"]),
        "synthetic_rbd_standby": standby,
        "synthetic_rbd_standby_mc_err": abs(standby - mc["standby"]),
        "synthetic_rbd_par_over_ser": float(parallel / series - 1.0),
    }
