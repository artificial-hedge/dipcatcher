"""Lagrangian relaxation + subgradient for set cover (SYNTHETIC).

min c'x, A x >= 1, x in {0,1}. Relaxing the coverage constraints with
multipliers lam >= 0 gives the separable lower bound
L(lam) = sum_i lam_i + sum_j min(0, c_j - lam' A_:j), maximized by
projected subgradient ascent. Bench reports the best bound vs the
brute-force ILP optimum and the bound-tightening gain over lam=0.
"""

import itertools

import numpy as np

from quant_fund.models._ilp_synth import SC_A, SC_C


def _lower_bound(lam: np.ndarray) -> tuple[float, np.ndarray]:
    red = SC_C - lam @ SC_A
    x = (red < 0).astype(float)
    return float(lam.sum() + np.minimum(red, 0).sum()), x


def _subgradient(iters: int = 600) -> tuple[float, int]:
    lam = np.zeros(SC_A.shape[0])
    best = 0.0
    best_it = 0
    for it in range(iters):
        lb, x = _lower_bound(lam)
        if lb > best:
            best, best_it = lb, it
        viol = 1.0 - SC_A @ x  # subgradient of L
        step = 2.0 / (it + 4)
        lam = np.maximum(0.0, lam + step * viol)
    return best, best_it


def _brute() -> float:
    best = np.inf
    for xs in itertools.product([0, 1], repeat=len(SC_C)):
        x = np.array(xs, dtype=float)
        if (SC_A @ x >= 1.0 - 1e-9).all():
            best = min(best, float(SC_C @ x))
    return best


def bench_lagrangian_relax(seed: int = 5107) -> dict[str, float]:
    lb0, _ = _lower_bound(np.zeros(SC_A.shape[0]))
    lb, it_best = _subgradient()
    truth = _brute()
    return {
        "synthetic_lag_lb0": lb0,
        "synthetic_lag_lb": lb,
        "synthetic_lag_truth": truth,
        "synthetic_lag_gap": float(truth - lb),
        "synthetic_lag_gap_frac": float((truth - lb) / truth),
        "synthetic_lag_best_iter": float(it_best),
    }
