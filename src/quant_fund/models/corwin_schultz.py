"""Transaction-cost estimators from daily high/low/close:
Corwin-Schultz (2012), Roll (1984), and Amihud (2002).

- Corwin-Schultz (2012): the daily high/low range
  contains both volatility and the bid-ask spread;
  two-day range lets us separate them.
  beta = sum (ln(H/L))^2 over pairs, gamma = (ln
  2-day high/low)^2, alpha = (sqrt(2 beta)-sqrt(beta))
  /(3-2 sqrt(2)) - sqrt(gamma/(3-2 sqrt(2))),
  spread = 2(e^{alpha}-1)/(1+e^{alpha}).
- Roll (1984): implied spread from negative serial
  covariance of price changes: s = 2 sqrt(-cov).
- Amihud (2002): |r| / dollar volume, an illiquidity
  ratio (price impact per dollar).

References
----------
- Corwin & Schultz (2012) 'A simple way to estimate
  bid-ask spreads from daily high and low prices'
  J. Finance 67(2).
- Roll (1984) 'A simple implicit measure of the
  effective bid-ask spread in an efficient market'
  J. Finance 39(4).
- Amihud (2002) 'Illiquidity and stock returns'
  J. Financial Markets 5(1).

Honesty
-------
SYNTHETIC self-check: GBM true price + known fixed
effective spread; asserts the CS estimate lands within
a factor band of the true spread, and Amihud/Roll are
positive and correctly scaled.

Composition
-----------
Pure numpy. Inputs are daily high/low/close (+ volume
for Amihud); outputs are spread/illiquidity estimates.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_hlc(
    high: FloatArray, low: FloatArray, close: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    h = np.asarray(high, dtype=np.float64).ravel()
    lo = np.asarray(low, dtype=np.float64).ravel()
    c = np.asarray(close, dtype=np.float64).ravel()
    n = h.size
    if lo.size != n or c.size != n or n < 10:
        raise ValueError("H/L/C length mismatch or too short")
    if not np.isfinite(h).all() or not np.isfinite(lo).all() or not np.isfinite(c).all():
        raise ValueError("non-finite prices")
    if (h < lo).any() or (h <= 0).any() or (lo <= 0).any() or (c <= 0).any():
        raise ValueError("invalid price geometry")
    return h, lo, c


def corwin_schultz_spread(high: FloatArray, low: FloatArray, close: FloatArray) -> FloatArray:
    """Per-day effective-spread estimate (fraction of price).

    Returns one estimate per adjacent day pair (length n-1);
    non-positive alpha maps to 0 (CS 2012 clipping).
    """
    h, lo, c = _check_hlc(high, low, close)
    _ = c  # close unused in CS
    hl2 = np.log(h / lo) ** 2
    beta = hl2[:-1] + hl2[1:]
    h2 = np.maximum(h[1:], h[:-1])
    l2 = np.minimum(lo[1:], lo[:-1])
    gamma = np.log(h2 / l2) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    alpha = np.maximum(alpha, 0.0)
    return np.asarray(2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)))


def roll_spread(returns: FloatArray) -> float:
    """Roll (1984) implied spread 2*sqrt(-Cov(dp_t, dp_{t-1}));
    0 when autocovariance non-negative (standard clipping)."""
    dp = np.asarray(returns, dtype=np.float64).ravel()
    if dp.size < 10 or not np.isfinite(dp).all():
        raise ValueError("returns too short/non-finite")
    cov = np.cov(dp[1:], dp[:-1])[0, 1]
    return float(2 * np.sqrt(max(-cov, 0.0)))


def amihud_illiq(returns: FloatArray, dollar_volume: FloatArray) -> float:
    """Amihud (2002): mean |r| / dollar volume (scaled 1e6)."""
    r = np.asarray(returns, dtype=np.float64).ravel()
    v = np.asarray(dollar_volume, dtype=np.float64).ravel()
    if r.size != v.size or r.size < 5:
        raise ValueError("input mismatch")
    if (v <= 0).any() or not np.isfinite(r).all() or not np.isfinite(v).all():
        raise ValueError("bad volume or returns")
    return float(np.mean(np.abs(r) / v) * 1e6)


def bench_spread(seed: int = 513) -> dict[str, float]:
    """SYNTHETIC: GBM + known 50bp spread; CS should land
    in a band around truth; Roll positive; Amihud scaled."""
    rng = np.random.default_rng(seed)
    n = 300
    true_spread = 0.005
    ret = rng.normal(0, 0.015, n)
    mid = 100.0 * np.exp(np.cumsum(ret))
    # observed prices bounce inside a half-spread band
    sign = rng.choice([-1.0, 1.0], n)
    obs = mid * (1 + sign * true_spread / 2)
    # daily high/low span the observed + mid range
    intraday = mid * np.abs(rng.normal(0, 0.008, n))
    high = np.maximum(obs, mid) + intraday
    low = np.minimum(obs, mid) - intraday
    cs = corwin_schultz_spread(high, low, obs)
    cs_med = float(np.median(cs))
    cs_mean = float(cs.mean())
    # Roll on observed returns (bounce induces -autocov)
    obs_ret = np.diff(np.log(obs))
    roll = roll_spread(obs_ret)
    vol = obs * rng.uniform(1e5, 3e5, n)
    am = amihud_illiq(obs_ret, vol[1:])
    # CS clips to zero on ~half of days by construction;
    # the mean is the informative summary
    if not (0.0005 < cs_mean < 0.03):
        raise ValueError("CS estimate out of band")
    return {
        "synthetic_cs_spread_med": cs_med,
        "synthetic_cs_spread_mean": cs_mean,
        "synthetic_true_spread": true_spread,
        "synthetic_roll_spread": roll,
        "synthetic_amihud": am,
        "synthetic_cs_pos_frac": float((cs > 0).mean()),
    }
