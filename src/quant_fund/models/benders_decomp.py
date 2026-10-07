"""Benders decomposition for uncapacitated facility location.

min f'y + q(y), q(y) = sum_i min_{j: y_j=1} c_ij. Combinatorial
(L-shaped) optimality cuts: at visited y_hat the master gains
z >= q(y_hat) - M * sum_j |y_j - y_hat_j| with M = max cost, a valid
lower bound since changing any open/closed site can reduce q by at
most M per flip. Master is enumerated over the 2^5 site subsets;
converged objective is checked against brute-force enumeration.
"""

import itertools

import numpy as np

from quant_fund.models._ilp_synth import FL_C, FL_F

# one site flip can cut each customer's assignment cost by at most
# max_ij c_ij, so total q change per flip is bounded by m * max_c
_M = float(FL_C.shape[0] * FL_C.max())


def _q(y: np.ndarray) -> float:
    total = 0.0
    for i in range(FL_C.shape[0]):
        total += min(FL_C[i, j] for j in range(len(y)) if y[j] > 0.5)
    return total


def _benders(iters: int = 40) -> tuple[float, int]:
    n = len(FL_F)
    cuts: list[tuple[np.ndarray, float]] = []
    best_obj = np.inf
    for _ in range(iters):
        best_master, y_star = np.inf, None
        for subset in itertools.product([0, 1], repeat=n):
            if not any(subset):
                continue
            y = np.array(subset, dtype=float)
            z = 0.0
            if cuts:
                z = max(qhat - _M * float(np.abs(y - yh).sum()) for yh, qhat in cuts)
            val = float(FL_F @ y) + z
            if val < best_master:
                best_master, y_star = val, y
        if not (y_star is not None):
            raise ValueError("y_star is not None")
        q_val = _q(y_star)
        obj = float(FL_F @ y_star) + q_val
        best_obj = min(best_obj, obj)
        if best_master >= best_obj - 1e-9:
            break
        cuts.append((y_star, q_val))
    return best_obj, len(cuts)


def _brute() -> float:
    n = len(FL_F)
    best = np.inf
    for subset in itertools.product([0, 1], repeat=n):
        if not any(subset):
            continue
        y = np.array(subset, dtype=float)
        best = min(best, float(FL_F @ y) + _q(y))
    return best


def bench_benders_decomp(seed: int = 5105) -> dict[str, float]:
    obj, n_cuts = _benders()
    truth = _brute()
    return {
        "synthetic_ben_obj": obj,
        "synthetic_ben_truth": truth,
        "synthetic_ben_gap": abs(obj - truth),
        "synthetic_ben_cuts": float(n_cuts),
    }
