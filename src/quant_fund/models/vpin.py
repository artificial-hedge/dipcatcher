"""VPIN — volume-synchronized probability of informed trading.

References
----------
- Easley, D., Lopez de Prado, M.M. & O'Hara, M. (2012). "Flow Toxicity
  and Liquidity in a High-frequency World." *Review of Financial
  Studies* 25(5), 1457-1493.
- Easley, D., Lopez de Prado, M.M. & O'Hara, M. (2011). "The
  Microstructure of the 'Flash Crash': Flow Toxicity, Liquidity
  Crashes, and the Probability of Informed Trading." *Journal of
  Portfolio Management* 37(2), 118-128.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence. VPIN is a flow-imbalance diagnostic, not
a return forecast.

Composition notes
-----------------
Distinct from ``pin_model`` (wave 44): EHO-PIN is a mixture-likelihood
estimate over daily buy/sell counts; VPIN is a nonparametric
volume-clock procedure — the tape is segmented into equal-volume
buckets, each bucket's buy share is bulk-classified as

    V_buy = V * Phi((p_t - p_{t-1}) / sigma_dp)

and toxicity over a rolling window of n buckets is

    VPIN = mean_i |V_buy_i - V_sell_i| / V_bucket.

The synthetic generator injects informed bursts (persistent signed
price moves inside a volume spike); the rolling VPIN CDF should flag
them against a noise-only baseline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def vpin(
    price: FloatArray,
    volume: FloatArray,
    n_buckets: int = 50,
    window: int = 10,
) -> dict[str, float]:
    """Volume-clock toxicity series summary.

    Buckets of ~equal volume are formed sequentially; per-bucket order
    imbalance is the bulk-classified |2*buy - 1| share. The returned
    ``vpin_mean`` is the rolling-mean toxicity over the last ``window``
    buckets and ``vpin_max`` its max.
    """
    pp = np.asarray(price, dtype=np.float64)
    vv = np.asarray(volume, dtype=np.float64)
    if pp.ndim != 1 or vv.ndim != 1 or pp.shape != vv.shape:
        raise ValueError("price and volume must be same-length series")
    n = pp.shape[0]
    if n < 100:
        raise ValueError("need n >= 100")
    if not np.all(np.isfinite(pp)) or not np.all(np.isfinite(vv)):
        raise ValueError("non-finite inputs")
    if np.any(vv <= 0) or np.any(pp <= 0):
        raise ValueError("price/volume must be positive")
    if n_buckets < 10 or window < 2:
        raise ValueError("n_buckets/window out of range")

    v_target = float(vv.sum()) / n_buckets
    dp = np.diff(pp)
    sd = float(np.std(dp, ddof=1))
    if sd <= 0:
        raise ValueError("degenerate price path")

    # Bulk-volume classification per trade, then aggregate into
    # equal-volume buckets.
    buy_frac = norm.cdf(dp / sd)
    buy_frac = np.concatenate([[0.5], buy_frac])
    buy_vol = vv * buy_frac
    sell_vol = vv * (1.0 - buy_frac)

    imbalances = []
    acc_b = 0.0
    acc_s = 0.0
    acc_v = 0.0
    for i in range(n):
        take = vv[i]
        # Fill the current bucket, possibly spilling over.
        while take > 0:
            space = v_target - acc_v
            chunk = min(take, space)
            acc_b += buy_vol[i] * (chunk / vv[i])
            acc_s += sell_vol[i] * (chunk / vv[i])
            acc_v += chunk
            take -= chunk
            if acc_v >= v_target - 1e-12:
                imbalances.append(abs(acc_b - acc_s) / v_target)
                acc_b = acc_s = acc_v = 0.0

    imb = np.asarray(imbalances)
    if imb.shape[0] < window:
        raise ValueError("too few buckets for window")
    roll = np.convolve(imb, np.ones(window) / window, mode="valid")
    return {
        "vpin_mean": float(np.mean(roll)),
        "vpin_max": float(np.max(roll)),
        "vpin_last": float(roll[-1]),
        "n_filled": float(imb.shape[0]),
    }


def vpin_cdf_level(series: FloatArray, value: float) -> float:
    """Empirical CDF of ``value`` under the VPIN series ``series``."""
    s = np.asarray(series, dtype=np.float64)
    if s.ndim != 1 or s.shape[0] < 5 or not np.all(np.isfinite(s)):
        raise ValueError("bad series")
    return float(np.mean(s <= value))


def synth_vpin(
    n: int = 4000,
    seed: int = 20261231 + 279,
    bursts: int = 6,
) -> dict[str, FloatArray]:
    """Noise tape with ``bursts`` informed-volume episodes.

    Baseline: symmetric noise trades (imbalance ~ 0). Each burst injects
    a signed price drift with 3x volume — bucket imbalances spike.
    """
    rng = np.random.default_rng(seed)
    if n < 100:
        raise ValueError("n too small")
    ret = rng.normal(0.0, 0.01, n)
    vol = rng.uniform(0.5, 1.5, n)
    loc = rng.choice(np.arange(200, n - 200), size=bursts, replace=False)
    sign = rng.choice([-1.0, 1.0], size=bursts)
    flag = np.zeros(n)
    for i, lo in enumerate(loc):
        w = 30
        ret[lo : lo + w] += sign[i] * 0.02
        vol[lo : lo + w] *= 3.0
        flag[lo : lo + w] = 1.0
    price = 100.0 * np.exp(np.cumsum(ret))
    return {"price": price, "volume": vol, "burst_flag": flag}


def bench_vpin(seed: int = 20261231 + 279) -> dict[str, float]:
    """Wave-48 self-check: VPIN on the burst tape exceeds the quiet-tape
    level by a clear margin."""
    d = synth_vpin(seed=seed)
    pp = np.asarray(d["price"])
    vv = np.asarray(d["volume"])
    a = vpin(pp, vv)
    # Quiet baseline: same generator without bursts.
    rng_q = np.random.default_rng(seed + 999)
    ret_q = rng_q.normal(0.0, 0.01, 4000)
    vol_q = rng_q.uniform(0.5, 1.5, 4000)
    p_q = 100.0 * np.exp(np.cumsum(ret_q))
    b = vpin(p_q, vol_q)
    a2 = vpin(pp, vv)
    detects = float(a["vpin_max"] > b["vpin_mean"] * 1.3)
    return {
        "synthetic_detects": detects,
        "synthetic_determinism": float(a == a2),
        "synthetic_vpin_max": a["vpin_max"],
        "synthetic_vpin_mean": a["vpin_mean"],
        "synthetic_quiet_mean": b["vpin_mean"],
        "synthetic_quiet_max": b["vpin_max"],
        "synthetic_n_filled": a["n_filled"],
    }
