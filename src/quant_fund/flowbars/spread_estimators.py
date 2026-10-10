"""Effective-spread estimators from price data.

- ``roll_implied_spread`` — Roll (1984): the implied half-spread from the
  negative lag-1 autocovariance of price changes, √max(−cov, 0);
- ``roll_implied_spread_full`` — the full bid/ask midpoint reconstruction
  (spread AND implied serial correlation of the efficient price), with the
  censoring that makes the estimator undefined when autocov ≥ 0;
- ``effective_spread_from_quotes`` — quoted half-spread averaged over a
  panel (the benchmark the estimators are checked against on synthetic
  quote data).

Honesty: spread estimators are noisy on short samples; the synthetic tests
use long samples where the estimand is known.

References:
- Roll, R. (1984). A simple implicit measure of the effective bid-ask
  spread — the covariance estimator.
- Hasbrouck, J. (2009). Trading costs and returns for US equities —
  effective vs quoted spreads.

Composition: pure numpy; deterministic; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _price_changes(prices: FloatArray) -> FloatArray:
    p = np.asarray(prices, dtype=np.float64)
    if p.ndim != 1 or len(p) < 10:
        raise ValueError("prices must be one-dimensional with >= 10 points")
    if np.any(p <= 0):
        raise ValueError("prices must be positive")
    return np.diff(p)


def roll_implied_spread(prices: FloatArray) -> float:
    """2·√max(−γ₁, 0) — the implied effective spread (price units)."""
    dp = _price_changes(prices)
    gamma1 = float(np.mean(dp[1:] * dp[:-1]) - np.mean(dp[1:]) * np.mean(dp[:-1]))
    return float(2.0 * np.sqrt(max(-gamma1, 0.0)))


def roll_implied_spread_full(prices: FloatArray) -> dict[str, float]:
    """Roll decomposition: spread, implied efficient-price variance, validity.

    Under Roll's model Δp_t = u_t + ε_t − ε_{t−1} with trade-sign imbalance
    c = P(buy) − P(sell): spread = 2√(−cov) and the implied variance of u is
    var(Δp) + 2cov. If lag-1 autocov ≥ 0 the estimator is undefined and the
    result reports valid=False with a zero spread.
    """
    dp = _price_changes(prices)
    mu = float(np.mean(dp))
    var_dp = float(np.mean((dp - mu) ** 2))
    gamma1 = float(np.mean((dp[1:] - mu) * (dp[:-1] - mu)))
    valid = gamma1 < 0
    spread = 2.0 * np.sqrt(-gamma1) if valid else 0.0
    eff_var = var_dp + 2.0 * gamma1 if valid else var_dp
    return {
        "spread": float(spread),
        "gamma1": gamma1,
        "efficient_var": max(eff_var, 0.0),
        "valid": 1.0 if valid else 0.0,
    }


def effective_spread_from_quotes(bid: FloatArray, ask: FloatArray) -> float:
    """Mean quoted half-spread (ask − bid)/2 — the benchmark for estimators."""
    b = np.asarray(bid, dtype=np.float64)
    a = np.asarray(ask, dtype=np.float64)
    if b.shape != a.shape or b.ndim != 1:
        raise ValueError("bid and ask must be one-dimensional arrays of equal shape")
    if np.any(a < b):
        raise ValueError("ask must be >= bid")
    return float(np.mean((a - b) / 2.0))
