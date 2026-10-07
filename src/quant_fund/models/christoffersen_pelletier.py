"""Christoffersen-Pelletier duration-based VaR backtest.

References
----------
- Christoffersen, P. & Pelletier, D. (2004). "Backtesting
  Value-at-Risk: A Duration-Based Approach." *Journal of
  Financial Econometrics* 2(1), 84-108.
- Christoffersen, P. (1998). "Evaluating Interval Forecasts."
  *International Economic Review* 39(4), 841-862.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
Given a hit sequence of VaR violations ``I_t in {0,1}``, the
durations between successive violations should be iid geometric
with success probability equal to the coverage level — which in
continuous-time is an exponential (constant-hazard) law.
Christoffersen-Pelletier fit a Weibull hazard to the censored
durations ``D_1..D_N`` (first duration censored on the left,
last on the right) and test ``b = 1`` (memoryless exponential
=> correct, no violation clustering) by the LR statistic
``LR = -2 (ln L_exp - ln L_weibull)`` against chi2(1). A
restricted-enrichment check: under violation clustering the
durations are over-dispersed (Weibull b < 1) and the LR
rejects; under an iid-coverage truth it does not. The module
reports ``durations``, MLE ``b_hat``, the LR, and the p-value,
and the synth plants (i) a correctly specified iid hit stream
with violations drawn at exactly the 5% rate and (ii) a
clustered two-regime stream whose durations fail the constant-
hazard test, requiring LR_reject > LR_accept materially.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as _opt
from scipy import stats as _stats

FloatArray = NDArray[np.float64]


def violation_durations(hits: FloatArray) -> FloatArray:
    """Durations between successive violations, censored ends."""
    hh = np.asarray(hits, dtype=np.float64)
    if hh.ndim != 1 or hh.size < 50 or not np.all(np.isfinite(hh)):
        raise ValueError("bad hits")
    idx = np.flatnonzero(hh > 0.5)
    if idx.size < 5:
        raise ValueError("too few violations")
    d = np.diff(idx)
    return np.concatenate([[float(idx[0] + 1)], d, [float(hh.size - idx[-1] - 1)]])


def _weibull_nll(b: float, d: FloatArray) -> float:
    """Negative log-likelihood of Weibull(shape=b) on censored durations.

    Interior durations are exact; first/last are right-censored
    (survival contribution) — using the scale a = T/N mean so
    the comparison with the exponential (b=1) is a restricted
    subfamily.
    """
    if b <= 0:
        return 1e12
    interior = d[1:-1]
    n = interior.size
    a = float(np.mean(d))
    ll = (
        n * (np.log(b) - b * np.log(a))
        + (b - 1.0) * np.sum(np.log(interior))
        - np.sum((interior / a) ** b)
    )
    # censored end durations contribute log S(d) = -(d/a)^b
    ll += -((d[0] / a) ** b) - ((d[-1] / a) ** b)
    return -float(ll)


def cp_backtest(hits: FloatArray) -> dict[str, float]:
    """Duration-based backtest: Weibull vs exponential hazard LR.

    Returns ``b_hat`` (Weibull shape MLE), ``lr`` (chi2(1)
    statistic for b=1), ``p_value``, ``n_durations`` and the
    violation rate.
    """
    d = violation_durations(hits)
    if np.any(d <= 0):
        raise ValueError("zero durations")
    a = float(np.mean(d))
    res = _opt.minimize_scalar(lambda b: _weibull_nll(b, d), bounds=(0.05, 5.0), method="bounded")
    b_hat = float(res.x)
    ll_w = -float(res.fun)
    ll_e = -_weibull_nll(1.0, d)
    lr = 2.0 * (ll_w - ll_e)
    p = 1.0 - float(_stats.chi2.cdf(lr, 1))
    _ = a
    return {
        "b_hat": b_hat,
        "lr": lr,
        "p_value": p,
        "n_durations": float(d.size),
        "hit_rate": float(np.mean(np.asarray(hits) > 0.5)),
    }


def synth_cp(
    seed: int = 20261231 + 309,
    t: int = 2500,
    alpha: float = 0.05,
    clustered: bool = False,
) -> FloatArray:
    """SYNTHETIC hit stream: iid at rate alpha or clustered regime."""
    rng = np.random.default_rng(seed)
    if not clustered:
        return (rng.uniform(size=t) < alpha).astype(np.float64)
    # clustered: Markov two-state violation propensity
    hits = np.zeros(t)
    state = 0
    p_stay = 0.98
    for i in range(t):
        if rng.uniform() < (p_stay if state else 1.0 - p_stay):
            pass
        else:
            state = 1 - state
        rate = 5.0 * alpha if state else 0.3 * alpha
        hits[i] = float(rng.uniform() < rate)
    return hits


def bench_christoffersen_pelletier(
    seed: int = 20261231 + 309,
) -> dict[str, float]:
    """Wave-53 self-check: iid passes, clustered rejected."""
    ok_hits = synth_cp(seed=seed, clustered=False)
    bad_hits = synth_cp(seed=seed, clustered=True)
    r_ok = cp_backtest(ok_hits)
    r_bad = cp_backtest(bad_hits)
    ok = r_ok["p_value"] > 0.01 and r_bad["p_value"] < 0.05
    return {
        "synthetic_b_iid": r_ok["b_hat"],
        "synthetic_p_iid": r_ok["p_value"],
        "synthetic_b_clustered": r_bad["b_hat"],
        "synthetic_p_clustered": r_bad["p_value"],
        "synthetic_lr_gap": r_bad["lr"] - r_ok["lr"],
        "synthetic_score": float(ok),
    }
