"""Bar-level feature engineering from OHLCV-style bar arrays.

Turns per-bar OHLC plus volume/dollar aggregates into aligned feature
arrays used by downstream microstructure diagnostics:

- ``bar_log_range`` / ``bar_body`` / ``bar_wick_asymmetry`` — shape features
  (all zero-safe and log1p-transformed where signed quantities could blow
  up);
- ``close_location_value`` — where the close sits inside the range in
  [−1, 1];
- ``bar_amihud`` — |return| / dollar volume per bar;
- ``bar_signed_volume`` — net signed volume per bar from tick-rule signs;
- ``bar_path_entropy`` — plug-in entropy of the intra-bar price path
  (requires per-trade prices; falls back to NaN when the bar has one trade);
- ``bar_feature_frame`` — compute all features at once into a dict of
  aligned float64 arrays.

Honesty: features are deterministic transforms of the supplied bars;
synthetic fixtures only.

References:
- López de Prado, M. (2018). *Advances in Financial Machine Learning*,
  ch. 2 — feature engineering on information-driven bars.
- Amihud, Y. (2002). Illiquidity and stock returns — the |ret|/$V ratio.

Composition: pure numpy; reuses quant_fund.flowbars.bars/entropy/imbalance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.flowbars.entropy import shannon_entropy
from quant_fund.flowbars.imbalance import tick_rule_signs

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _check_ohlc(ohlc: dict[str, FloatArray]) -> None:
    for key in ("open", "high", "low", "close"):
        if key not in ohlc:
            raise ValueError(f"ohlc missing key {key!r}")
        if ohlc[key].ndim != 1:
            raise ValueError(f"ohlc[{key!r}] must be one-dimensional")
    n = len(ohlc["open"])
    for key in ("high", "low", "close"):
        if len(ohlc[key]) != n:
            raise ValueError("ohlc arrays must have equal length")
    if np.any(ohlc["high"] < ohlc["low"]):
        raise ValueError("high must be >= low in every bar")
    if np.any(ohlc["close"] <= 0) or np.any(ohlc["open"] <= 0):
        raise ValueError("prices must be positive")


def bar_log_range(ohlc: dict[str, FloatArray]) -> FloatArray:
    """log(high/low) — the bar range on the log scale."""
    _check_ohlc(ohlc)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(ohlc["low"] > 0, np.log(ohlc["high"] / ohlc["low"]), np.nan)
    return np.asarray(out, dtype=np.float64)


def bar_body(ohlc: dict[str, FloatArray]) -> FloatArray:
    """log(close/open) — the signed bar body."""
    _check_ohlc(ohlc)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(ohlc["open"] > 0, np.log(ohlc["close"] / ohlc["open"]), np.nan)
    return np.asarray(out, dtype=np.float64)


def bar_wick_asymmetry(ohlc: dict[str, FloatArray]) -> FloatArray:
    """(upper − lower wick) / range in [−1, 1]; 0 when the range is zero.

    upper wick = high − max(open, close); lower wick = min(open, close) − low.
    """
    _check_ohlc(ohlc)
    up = ohlc["high"] - np.maximum(ohlc["open"], ohlc["close"])
    lo = np.minimum(ohlc["open"], ohlc["close"]) - ohlc["low"]
    rng = ohlc["high"] - ohlc["low"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(rng > 0, (up - lo) / rng, 0.0)
    return np.asarray(out, dtype=np.float64)


def close_location_value(ohlc: dict[str, FloatArray]) -> FloatArray:
    """(2·close − high − low) / (high − low) in [−1, 1]; 0 for zero-range bars."""
    _check_ohlc(ohlc)
    rng = ohlc["high"] - ohlc["low"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(rng > 0, (2.0 * ohlc["close"] - ohlc["high"] - ohlc["low"]) / rng, 0.0)
    return np.asarray(out, dtype=np.float64)


def bar_amihud(ohlc: dict[str, FloatArray], dollar_volume: FloatArray) -> FloatArray:
    """|close-to-close return| / dollar volume per bar (first bar = NaN)."""
    _check_ohlc(ohlc)
    dv = np.asarray(dollar_volume, dtype=np.float64)
    if dv.shape != ohlc["close"].shape:
        raise ValueError("dollar_volume must match the close array")
    close = ohlc["close"]
    ret = np.full(len(close), np.nan, dtype=np.float64)
    ret[1:] = np.abs(close[1:] / close[:-1] - 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(dv > 0, ret / dv, np.nan)
    return np.asarray(out, dtype=np.float64)


def bar_signed_volume(
    trade_prices: FloatArray,
    trade_sizes: FloatArray,
    bar_ids: IntArray,
) -> FloatArray:
    """Net signed volume per bar from tick-rule signs on the trade tape."""
    prices = np.asarray(trade_prices, dtype=np.float64)
    sizes = np.asarray(trade_sizes, dtype=np.float64)
    ids = np.asarray(bar_ids, dtype=np.int64)
    if not (prices.shape == sizes.shape == ids.shape):
        raise ValueError("trade_prices, trade_sizes and bar_ids must have equal shapes")
    signs = tick_rule_signs(prices)
    n_bars = int(ids[-1]) + 1 if len(ids) else 0
    out = np.zeros(n_bars, dtype=np.float64)
    np.add.at(out, ids, signs.astype(np.float64) * sizes)
    return out


def bar_path_entropy(
    trade_prices: FloatArray,
    bar_ids: IntArray,
    bins: int = 8,
) -> FloatArray:
    """Plug-in entropy of per-bar return paths; NaN for one-trade bars."""
    prices = np.asarray(trade_prices, dtype=np.float64)
    ids = np.asarray(bar_ids, dtype=np.int64)
    if prices.shape != ids.shape:
        raise ValueError("trade_prices and bar_ids must have equal shapes")
    n_bars = int(ids[-1]) + 1 if len(ids) else 0
    out = np.full(n_bars, np.nan, dtype=np.float64)
    for b in range(n_bars):
        p = prices[ids == b]
        if len(p) >= 4 and np.all(p > 0):
            r = np.diff(np.log(p))
            out[b] = shannon_entropy(r, bins=bins)["nats"]
    return out


def bar_feature_frame(
    ohlc: dict[str, FloatArray],
    dollar_volume: FloatArray,
    trade_prices: FloatArray | None = None,
    trade_sizes: FloatArray | None = None,
    bar_ids: IntArray | None = None,
    bins: int = 8,
) -> dict[str, FloatArray]:
    """All bar features in one dict of aligned float64 arrays.

    Signed-volume and path-entropy entries are only present when the trade
    tape is supplied.
    """
    frame = {
        "log_range": bar_log_range(ohlc),
        "body": bar_body(ohlc),
        "wick_asymmetry": bar_wick_asymmetry(ohlc),
        "close_location": close_location_value(ohlc),
        "amihud": bar_amihud(ohlc, dollar_volume),
    }
    if trade_prices is not None and trade_sizes is not None and bar_ids is not None:
        frame["signed_volume"] = bar_signed_volume(trade_prices, trade_sizes, bar_ids)
        frame["path_entropy"] = bar_path_entropy(trade_prices, bar_ids, bins=bins)
    return frame
