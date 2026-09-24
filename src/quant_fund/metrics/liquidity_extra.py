"""Additional liquidity and drawdown-adjusted performance ratios.

- **Hui-Heubel** (1984) liquidity ratio: the price range over a window relative
  to turnover, ``LR = ((P_max - P_min)/P_min) / (V / (S * P_bar))``; higher
  values mean less liquidity (price moves more per unit of turnover).
- **Martin ratio / Ulcer Performance Index** (Martin & McCann 1989): excess
  return divided by the Ulcer Index (the RMS drawdown), rewarding smooth
  equity curves.

Fail-closed on non-finite input or degenerate denominators.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def hui_heubel_ratio(prices: Array, volume_shares: Array, shares_outstanding: float) -> float:
    """Hui-Heubel liquidity ratio over the supplied window (higher = less liquid)."""
    p = np.asarray(prices, dtype=float).ravel()
    v = np.asarray(volume_shares, dtype=float).ravel()
    if p.size < 2 or p.size != v.size or not (np.isfinite(p).all() and np.isfinite(v).all()):
        raise ValueError("prices and volume must be finite, aligned, length >= 2")
    if shares_outstanding <= 0.0 or (p <= 0).any() or (v < 0).any():
        raise ValueError("require positive prices, shares outstanding, non-negative volume")
    p_min = float(p.min())
    price_range = (float(p.max()) - p_min) / p_min
    p_bar = float(p.mean())
    dollar_volume = float(np.sum(v * p))
    turnover = dollar_volume / (shares_outstanding * p_bar)
    if turnover <= 0.0:
        raise ValueError("zero turnover; ratio undefined")
    return float(price_range / turnover)


def ulcer_index(returns: Array) -> float:
    """Ulcer Index: RMS percentage drawdown of the equity curve."""
    r = np.asarray(returns, dtype=float).ravel()
    if r.size < 2 or not np.isfinite(r).all():
        raise ValueError("returns must be finite with >= 2 observations")
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    drawdown = 100.0 * (equity - peak) / peak
    return float(np.sqrt(np.mean(drawdown**2)))


def martin_ratio(returns: Array, rf: float = 0.0, periods_per_year: float = 252.0) -> float:
    """Martin ratio (Ulcer Performance Index): annualised excess return / Ulcer Index."""
    r = np.asarray(returns, dtype=float).ravel()
    if r.size < 2 or not np.isfinite(r).all():
        raise ValueError("returns must be finite with >= 2 observations")
    ui = ulcer_index(r)
    if ui <= 0.0:
        raise ValueError("zero Ulcer Index (no drawdown); ratio undefined")
    ann_excess = float(r.mean() - rf) * periods_per_year
    return float(ann_excess / ui)
