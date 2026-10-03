"""Epps-Singleton (1985) characteristic-function normality test.

One-sample omnibus normality test comparing the empirical
characteristic function (ECF) of the standardized sample
to the N(0,1) characteristic function at the canonical
frequencies t = {0.5, 1}. With g_j the cos/sin parts of
the ECF, the statistic

    G_1 = n * v' Sigma^{-1} v  ~  chi^2(4)

where v collects the four ECF deviations and Sigma is the
exact Gaussian covariance kernel

    cov(ce_a, ce_b) = 0.5 (phi(a+b) + phi(a-b)) - phi(a) phi(b)
    cov(se_a, se_b) = 0.5 (phi(a-b) - phi(a+b))
    cov(ce_a, se_b) = 0

with phi(t) = exp(-t^2/2). Powerful against skew and
heavy-tail alternatives where moment tests are weak.

Honesty: `bench_epps_singleton` checks the test accepts
Gaussian data and rejects a skewed lognormal and a
heavy-tailed t(3) sample on SYNTHETIC data — a size/power
diagnostic, not a real-data normality claim.

References
----------
* Epps & Singleton (1986) "An omnibus test for the two-
  sample problem using the empirical characteristic
  function", J Stat Comput Simul 26, 177-203.
* Epps (1987) "Testing that a stationary time series is
  Gaussian", Annals of Statistics 15(4).
* Baringhaus & Henze (1988) characteristic-function tests
  review, Metrika 35.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _phi(t: float) -> float:
    return math.exp(-0.5 * t * t)


def epps_singleton(x: FloatArray) -> dict[str, float]:
    """Epps-Singleton G_1 statistic + chi^2(4) p-value."""
    xx = np.asarray(x, dtype=np.float64).ravel()
    n = xx.size
    if n < 8 or not np.all(np.isfinite(xx)):
        raise ValueError("bad input")
    sd = float(xx.std(ddof=1))
    if sd <= 0:
        raise ValueError("degenerate sample")
    y = (xx - xx.mean()) / sd
    ts = (0.5, 1.0)
    ce = np.array([float(np.cos(t * y).mean()) for t in ts])
    se = np.array([float(np.sin(t * y).mean()) for t in ts])
    v = np.array([ce[0] - _phi(0.5), se[0], ce[1] - _phi(1.0), se[1]])

    # exact H0 covariance kernel on [ce(0.5), se(0.5), ce(1), se(1)]
    def k_cc(a: float, b: float) -> float:
        return 0.5 * (_phi(a + b) + _phi(a - b)) - _phi(a) * _phi(b)

    def k_ss(a: float, b: float) -> float:
        return 0.5 * (_phi(a - b) - _phi(a + b))

    cov = np.array(
        [
            [k_cc(0.5, 0.5), 0.0, k_cc(0.5, 1.0), 0.0],
            [0.0, k_ss(0.5, 0.5), 0.0, k_ss(0.5, 1.0)],
            [k_cc(1.0, 0.5), 0.0, k_cc(1.0, 1.0), 0.0],
            [0.0, k_ss(1.0, 0.5), 0.0, k_ss(1.0, 1.0)],
        ]
    )
    g1 = float(n * v @ np.linalg.solve(cov, v))
    p = float(stats.chi2.sf(g1, 4))
    return {"g1": g1, "p": p}


def bench_epps_singleton(seed: int = 20261231 + 471) -> dict[str, float]:
    """SYNTHETIC check — accepts N(0,1), rejects lognormal and t(3)."""
    rng = np.random.default_rng(seed)
    n = 800
    out_n = epps_singleton(rng.standard_normal(n))
    if out_n["p"] < 0.05:
        raise ValueError(f"rejected normal: p={out_n['p']}")
    out_ln = epps_singleton(rng.lognormal(size=n))
    if out_ln["p"] > 0.05:
        raise ValueError(f"accepted lognormal: p={out_ln['p']}")
    out_t = epps_singleton(rng.standard_t(3.0, size=n))
    if out_t["p"] > 0.05:
        raise ValueError(f"accepted t3: p={out_t['p']}")
    return {
        "synthetic_normal_p": float(out_n["p"]),
        "synthetic_lognormal_p": float(out_ln["p"]),
        "synthetic_t3_p": float(out_t["p"]),
        "score": 1.0,
    }
