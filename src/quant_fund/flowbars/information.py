"""Information (volatility-clock) bars.

Bars cut when the cumulative squared log-return since the bar open crosses a
threshold — the discrete analogue of sampling by quadratic variation. This
makes bars carry roughly equal *price-information* content regardless of how
fast the tape is moving, complementing dollar bars (equal value content) and
imbalance bars (equal signed-flow content).

Honesty: deterministic functions of the input tape; synthetic fixtures only.

References:
- López de Prado, M. (2018). *Advances in Financial Machine Learning*,
  ch. 2 — the information-driven bar family.
- Aït-Sahalia, Y., Jacod, J. (2014). *High-Frequency Financial Econometrics*
  — quadratic variation as the information clock.

Composition: pure numpy; deterministic; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def information_bar_ids(prices: FloatArray, threshold: float) -> IntArray:
    """Bars cut on cumulative squared log-returns ≥ threshold.

    The first trade opens bar 0 (no return observed yet); the final open bar
    is closed at the last trade.
    """
    prices = np.asarray(prices, dtype=np.float64)
    if prices.ndim != 1:
        raise ValueError("prices must be one-dimensional")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if np.any(prices <= 0):
        raise ValueError("prices must be positive for log-returns")
    n = len(prices)
    ids = np.zeros(n, dtype=np.int64)
    if n == 0:
        return ids
    log_p = np.log(prices)
    bar = 0
    run = 0.0
    for i in range(1, n):
        r = float(log_p[i] - log_p[i - 1])
        run += r * r
        if run >= threshold:
            bar += 1
            run = 0.0
        ids[i] = bar
    return ids


def cumulative_quadratic_variation(prices: FloatArray) -> FloatArray:
    """Running sum of squared log-returns (diagnostic; starts at 0)."""
    prices = np.asarray(prices, dtype=np.float64)
    if np.any(prices <= 0):
        raise ValueError("prices must be positive for log-returns")
    r = np.diff(np.log(prices))
    out = np.zeros(len(prices), dtype=np.float64)
    out[1:] = np.cumsum(r * r)
    return out
