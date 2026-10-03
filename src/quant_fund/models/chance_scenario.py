"""Scenario approach for a chance-constrained program.

min c'x s.t. P(A_omega x <= b_omega) >= 1-eps. Sample N iid
constraints, solve the deterministic problem on them; the scenario
bound eps ~ 2/(N+1) * (d + sqrt stuff) bounds the violation. Bench
reports the sampled solution's empirical violation rate vs the bound.
"""

import numpy as np
from scipy.optimize import linprog


def _sample_c(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    a = rng.uniform(0.5, 2.0, (n, 2))
    b = rng.uniform(1.0, 3.0, n)
    return a, b


def _eps_bound(n: int, d: int, beta: float = 0.01) -> float:
    """Campi-Garatti: eps with sum_{i<d} C(n,i) eps^i (1-eps)^(n-i) = beta."""
    from scipy.stats import binom

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom.cdf(d - 1, n, mid) > beta:
            lo = mid
        else:
            hi = mid
    return lo


def bench_chance_scenario(seed: int = 5507) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    c = np.array([1.0, 1.0])
    for n in (60, 200):
        a_s, b_s = _sample_c(rng, n)
        # covering form: min c'x s.t. a'x >= b for each sampled row
        res = linprog(c, A_ub=-a_s, b_ub=-b_s, bounds=(0, 20), method="highs")
        x = res.x
        a_t, b_t = _sample_c(rng, 200000)
        viol = float(np.mean(a_t @ x < b_t))
        bound = _eps_bound(n, 2)
        if n == 60:
            v60, b60 = viol, bound
        else:
            v200, b200 = viol, bound
    return {
        "synthetic_cc_viol_60": v60,
        "synthetic_cc_bound_60": b60,
        "synthetic_cc_viol_200": v200,
        "synthetic_cc_bound_200": b200,
        "synthetic_cc_shrinks": float(v200 < v60),
        "synthetic_cc_bound_shrinks": float(b200 < b60),
    }
