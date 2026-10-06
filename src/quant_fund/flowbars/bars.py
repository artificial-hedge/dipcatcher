"""Threshold-bar construction from a trade tape.

Dollar, volume, and tick bars cut the tape whenever cumulative traded value
(dollar bars) or quantity (volume bars) crosses a fixed threshold, instead of
sampling in clock time. The thresholding rule is::

    bar_id(t) = rank(floor(cumsum(metric)(t) / threshold))

which is vectorised, deterministic, and monotone in the threshold; the rank
compression keeps bar ids dense even when one trade crosses several
threshold multiples at once. Cutting on traded value makes each bar carry
roughly equal information content, which is the standard argument for
preferring these bars to time bars in microstructure studies.

Honesty: the bundled ``synth_tape`` generator is a geometric random walk with
random sizes — synthetic correctness fixtures only.

References:
- López de Prado, M. (2018). *Advances in Financial Machine Learning*,
  ch. 2 — dollar/volume/tick bars and why sampling in event time helps.
- Easley, D., López de Prado, M., O'Hara, M. (2012). Flow toxicity and
  liquidity in a high-frequency world — volume-clock reasoning.

Composition: pure numpy; deterministic ``np.random.default_rng``; no new
dependencies beyond the locked core.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def synth_tape(
    n_trades: int = 10_000,
    *,
    seed: int = 0,
    drift: float = 0.0,
    vol: float = 0.01,
    size_lambda: float = 10.0,
) -> dict[str, FloatArray]:
    """Synthetic trade tape: geometric price walk with Poisson sizes.

    Returns dict with ``price``, ``size`` (float64) and the derived
    ``dollar`` (price × size) per trade. Labeled synthetic — never market
    evidence.
    """
    if n_trades < 2:
        raise ValueError("n_trades must be >= 2")
    rng = np.random.default_rng(seed)
    sizes = rng.poisson(lam=size_lambda, size=n_trades).astype(np.float64) + 1.0
    shocks = rng.normal(loc=drift, scale=vol, size=n_trades)
    log_price = np.cumsum(shocks)
    prices = 100.0 * np.exp(log_price)
    return {
        "price": prices.astype(np.float64),
        "size": sizes,
        "dollar": (prices * sizes).astype(np.float64),
    }


def _threshold_bar_ids(cumulative: FloatArray, threshold: float) -> IntArray:
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if cumulative.ndim != 1:
        raise ValueError("cumulative must be one-dimensional")
    ids = np.floor(cumulative / threshold).astype(np.int64)
    # A single trade can cross several threshold multiples at once; the
    # intervening multiples correspond to bars with no trades. Compress to
    # dense rank ids so every bar id owns at least one trade — consistent
    # with the imbalance/run scanners, which are dense by construction.
    _, dense = np.unique(ids, return_inverse=True)
    return dense.astype(np.int64)


def dollar_bar_ids(dollar: FloatArray, threshold: float) -> IntArray:
    """Bar index per trade for dollar bars (cumulative $ value crossing)."""
    return _threshold_bar_ids(np.cumsum(np.asarray(dollar, dtype=np.float64)), threshold)


def volume_bar_ids(sizes: FloatArray, threshold: float) -> IntArray:
    """Bar index per trade for volume bars (cumulative quantity crossing)."""
    return _threshold_bar_ids(np.cumsum(np.asarray(sizes, dtype=np.float64)), threshold)


def tick_bar_ids(n_trades: int, every: int) -> IntArray:
    """Bar index per trade for tick bars (one bar per ``every`` trades)."""
    if n_trades < 0:
        raise ValueError("n_trades must be non-negative")
    if every <= 0:
        raise ValueError("every must be positive")
    return (np.arange(n_trades, dtype=np.int64) // every).astype(np.int64)


def _group_ends(ids: IntArray) -> IntArray:
    """Index of the last trade of each bar (ids must be sorted)."""
    change = np.flatnonzero(np.diff(ids))
    ends = np.append(change, len(ids) - 1)
    return ends.astype(np.int64)


def bar_ohlc(prices: FloatArray, ids: IntArray) -> dict[str, FloatArray]:
    """OHLC per bar from per-trade prices and bar ids (sorted)."""
    prices = np.asarray(prices, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    if prices.shape != ids.shape:
        raise ValueError("prices and ids must have the same shape")
    if len(prices) == 0:
        return {"open": np.empty(0), "high": np.empty(0), "low": np.empty(0), "close": np.empty(0)}
    ends = _group_ends(ids)
    starts = np.concatenate(([0], ends[:-1] + 1))
    out_open = prices[starts]
    out_close = prices[ends]
    out_high = np.maximum.reduceat(prices, starts)
    out_low = np.minimum.reduceat(prices, starts)
    return {
        "open": np.asarray(out_open, dtype=np.float64),
        "high": np.asarray(out_high, dtype=np.float64),
        "low": np.asarray(out_low, dtype=np.float64),
        "close": np.asarray(out_close, dtype=np.float64),
    }


def bar_last_prices(prices: FloatArray, ids: IntArray) -> FloatArray:
    """Closing price of each bar (last trade price inside the bar)."""
    prices = np.asarray(prices, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    if prices.shape != ids.shape:
        raise ValueError("prices and ids must have the same shape")
    if len(prices) == 0:
        return np.empty(0, dtype=np.float64)
    return np.asarray(prices[_group_ends(ids)], dtype=np.float64)


def bar_returns(prices: FloatArray, ids: IntArray) -> FloatArray:
    """Simple returns between consecutive bar close prices."""
    closes = bar_last_prices(prices, ids)
    if len(closes) < 2:
        return np.empty(0, dtype=np.float64)
    prev = closes[:-1]
    curr = closes[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(prev != 0.0, curr / prev - 1.0, np.nan)
    return np.asarray(out, dtype=np.float64)
