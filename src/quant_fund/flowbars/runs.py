"""Runs bars: cut the tape on streaks, not magnitude.

A runs bar closes when the cumulative count of same-sign flow in the
*dominant* direction (buy-run vs sell-run) reaches the threshold. Unlike
imbalance bars (net flow), runs bars react to persistence of one side of
the market, so they fire when order flow becomes one-sided even if buys and
sells partially cancel.

Honesty: deterministic functions of the input tape; synthetic fixtures only.

References:
- López de Prado, M. (2018). *Advances in Financial Machine Learning*,
  ch. 2 — tick/volume/dollar runs bars.
- Cont, R. (2001). Empirical properties of asset returns: stylized facts
  and statistical issues — one-sided flow episodes.

Composition: pure numpy; deterministic; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def run_bar_ids(signs: IntArray, values: FloatArray, threshold: float) -> IntArray:
    """Generic runs bars: close when max(buy-run, sell-run) ≥ threshold.

    ``values`` weights each trade (1 → tick runs, sizes → volume runs,
    dollars → dollar runs). The final open bar is closed at the last trade.
    """
    signs = np.asarray(signs, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if signs.shape != values.shape:
        raise ValueError("signs and values must have the same shape")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    n = len(signs)
    ids = np.zeros(n, dtype=np.int64)
    if n == 0:
        return ids
    buy_run = 0.0
    sell_run = 0.0
    bar = 0
    for i in range(n):
        v = float(values[i])
        if signs[i] > 0:
            buy_run += v
            sell_run = 0.0
        else:
            sell_run += v
            buy_run = 0.0
        ids[i] = bar
        if max(buy_run, sell_run) >= threshold:
            bar += 1
            buy_run = 0.0
            sell_run = 0.0
    return ids


def tick_run_bar_ids(signs: IntArray, threshold: float) -> IntArray:
    """Tick runs bars: consecutive same-sign trades, each counting 1."""
    signs = np.asarray(signs, dtype=np.int64)
    return run_bar_ids(signs, np.ones(len(signs), dtype=np.float64), threshold)


def volume_run_bar_ids(signs: IntArray, sizes: FloatArray, threshold: float) -> IntArray:
    """Volume runs bars: consecutive same-sign trades weighted by size."""
    return run_bar_ids(signs, sizes, threshold)


def dollar_run_bar_ids(signs: IntArray, dollars: FloatArray, threshold: float) -> IntArray:
    """Dollar runs bars: consecutive same-sign trades weighted by dollars."""
    return run_bar_ids(signs, dollars, threshold)


def max_run_profile(signs: IntArray, values: FloatArray) -> FloatArray:
    """Per-trade maximum open same-sign run at that point (diagnostic)."""
    signs = np.asarray(signs, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if signs.shape != values.shape:
        raise ValueError("signs and values must have the same shape")
    n = len(signs)
    out = np.zeros(n, dtype=np.float64)
    buy_run = 0.0
    sell_run = 0.0
    for i in range(n):
        v = float(values[i])
        if signs[i] > 0:
            buy_run += v
            sell_run = 0.0
        else:
            sell_run += v
            buy_run = 0.0
        out[i] = max(buy_run, sell_run)
    return out
