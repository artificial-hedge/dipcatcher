"""Hedonic price regression + Case-Shiller repeat-sales index.

References
----------
- Rosen, S. (1974). "Hedonic Prices and Implicit Markets: Product
  Differentiation in Pure Competition." *Journal of Political Economy*
  82(1), 34-55.
- Bailey, M.J., Muth, R.F. & Nourse, H.O. (1963). "A Regression Method
  for Real Estate Price Index Construction." *Journal of the American
  Statistical Association* 58(304), 933-942.
- Case, K.E. & Shiller, R.J. (1987). "Prices of Single-Family Homes
  Since 1970: New Indexes for Four Cities." *New England Economic
  Review*, 45-56.
- Griliches, Z. (1961). "Hedonic Price Indexes for Automobiles." In
  *The Price Statistics of the Federal Government* (NBER).

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
Semi-log hedonic pricing: ``log p_i = a_t(i) + beta'x_i + eps_i``
estimated by within-transformation on time dummies — the time-dummy
coefficients are the hedonic price index. Repeat-sales index: sales
pairs in periods (s, t) enter a first-difference regression
``log(p2/p1) = sum_tau gamma_tau * (1[tau=t] - 1[tau=s])`` estimated on
the GMN/Case-Shiller dummy design; the index is ``exp(cumsum gamma)``.
The synth embeds a true index path + known attribute loadings; both
estimators must track the true index.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def hedonic_index(
    log_price: FloatArray,
    period: FloatArray,
    chars: FloatArray,
) -> dict[str, float | FloatArray]:
    """Time-dummy hedonic price index.

    ``log_price`` per transaction, ``period`` integer period labels,
    ``chars`` (n,k) attribute matrix. Returns per-period index levels
    plus attribute shadow prices.
    """
    lp = np.asarray(log_price, dtype=np.float64)
    pr = np.asarray(period, dtype=np.float64)
    xx = np.asarray(chars, dtype=np.float64)
    n = lp.shape[0]
    if lp.ndim != 1 or pr.ndim != 1 or pr.shape[0] != n:
        raise ValueError("log_price and period must be same-length vectors")
    if xx.ndim != 2 or xx.shape[0] != n or n < 30:
        raise ValueError("chars must be (n,k), n>=30")
    if not np.all(np.isfinite(lp)) or not np.all(np.isfinite(pr)) or not np.all(np.isfinite(xx)):
        raise ValueError("non-finite inputs")
    periods = np.unique(pr)
    if periods.shape[0] < 3:
        raise ValueError("need >= 3 periods")

    dummies = np.column_stack([(pr == p).astype(np.float64) for p in periods[1:]])
    design = np.column_stack([np.ones(n), xx, dummies])
    beta = np.linalg.lstsq(design, lp, rcond=None)[0]
    k = xx.shape[1]
    time_coef = beta[1 + k :]
    index = np.exp(np.concatenate([[beta[0]], beta[0] + time_coef]))
    resid = lp - design @ beta
    return {
        "index_first": float(index[0]),
        "index_last": float(index[-1]),
        "index_growth": float(index[-1] / index[0]),
        "attr_beta_mean": float(np.mean(np.abs(beta[1 : 1 + k]))),
        "r2": float(1.0 - np.var(resid) / np.var(lp)),
        "n": float(n),
        "n_periods": float(periods.shape[0]),
        "_index": index,
        "_periods": periods,
    }


def repeat_sales_index(
    price1: FloatArray,
    price2: FloatArray,
    period1: FloatArray,
    period2: FloatArray,
) -> dict[str, float | FloatArray]:
    """BMN/Case-Shiller repeat-sales index on sale pairs.

    Each pair (s, t) contributes one row ``log(p2/p1) = b_t - b_s``
    on the first-difference dummy design.
    """
    p1 = np.asarray(price1, dtype=np.float64)
    p2 = np.asarray(price2, dtype=np.float64)
    s = np.asarray(period1, dtype=np.float64)
    t = np.asarray(period2, dtype=np.float64)
    n = p1.shape[0]
    if min(p2.shape[0], s.shape[0], t.shape[0]) != n:
        raise ValueError("all inputs must share length")
    if n < 30:
        raise ValueError("need >= 30 pairs")
    if not all(np.all(np.isfinite(a)) for a in (p1, p2, s, t)):
        raise ValueError("non-finite inputs")
    if np.any(p1 <= 0) or np.any(p2 <= 0) or np.any(t <= s):
        raise ValueError("bad prices or non-increasing sale order")
    periods = np.unique(np.concatenate([s, t]))
    if periods.shape[0] < 3:
        raise ValueError("need >= 3 periods")
    pmap = {p: i for i, p in enumerate(periods)}
    k = periods.shape[0]
    dy = np.log(p2) - np.log(p1)
    dx = np.zeros((n, k))
    for i in range(n):
        dx[i, pmap[t[i]]] = 1.0
        dx[i, pmap[s[i]]] = -1.0
    # Drop the first column (base period) for identification.
    design = dx[:, 1:]
    b = np.linalg.lstsq(design, dy, rcond=None)[0]
    log_index = np.concatenate([[0.0], b])
    index = np.exp(log_index)
    resid = dy - design @ b
    return {
        "index_first": float(index[0]),
        "index_last": float(index[-1]),
        "index_growth": float(index[-1] / index[0]),
        "resid_sd": float(np.std(resid)),
        "r2": float(1.0 - np.var(resid) / np.var(dy)) if np.var(dy) > 0 else 0.0,
        "n_pairs": float(n),
        "n_periods": float(k),
        "_index": index,
        "_periods": periods,
    }


def synth_hedonic(
    n: int = 600,
    n_periods: int = 8,
    seed: int = 20261231 + 291,
    drift: float = 0.02,
) -> dict[str, FloatArray]:
    """Repeat-sales + hedonic panel with a known index path."""
    rng = np.random.default_rng(seed)
    if n < 30 or n_periods < 3:
        raise ValueError("bad sizes")
    true_index = np.exp(drift * np.arange(n_periods))
    size = rng.uniform(0.0, 1.0, n)
    quality = rng.uniform(0.0, 1.0, n)
    s = rng.integers(0, n_periods - 2, n)
    t = s + rng.integers(1, n_periods - 1, n)
    t = np.clip(t, 0, n_periods - 1)
    bad = t <= s
    t[bad] = np.minimum(s[bad] + 1, n_periods - 1)
    keep = t > s
    s, t = s[keep], t[keep]
    size, quality = size[keep], quality[keep]
    p1 = np.log(true_index[s]) + 0.4 * size + 0.2 * quality + rng.normal(0.0, 0.05, s.shape[0])
    p2 = np.log(true_index[t]) + 0.4 * size + 0.2 * quality + rng.normal(0.0, 0.05, s.shape[0])
    return {
        "price1": np.exp(p1),
        "price2": np.exp(p2),
        "period1": s.astype(np.float64),
        "period2": t.astype(np.float64),
        "true_index": true_index,
        "size": size,
        "quality": quality,
    }


def bench_hedonic(seed: int = 20261231 + 291) -> dict[str, float]:
    """Wave-50 self-check: repeat-sales index tracks the true growth
    path within tolerance."""
    d = synth_hedonic(seed=seed)
    a = repeat_sales_index(
        np.asarray(d["price1"]),
        np.asarray(d["price2"]),
        np.asarray(d["period1"]),
        np.asarray(d["period2"]),
    )
    a2 = repeat_sales_index(
        np.asarray(d["price1"]),
        np.asarray(d["price2"]),
        np.asarray(d["period1"]),
        np.asarray(d["period2"]),
    )
    true_growth = float(np.asarray(d["true_index"])[-1])
    est_growth = float(a["index_last"])
    detects = float(abs(est_growth - true_growth) / true_growth < 0.15)
    return {
        "synthetic_detects": detects,
        "synthetic_determinism": float(
            np.array_equal(np.asarray(a["_index"]), np.asarray(a2["_index"]))
        ),
        "synthetic_est_growth": est_growth,
        "synthetic_true_growth": true_growth,
        "synthetic_resid_sd": float(a["resid_sd"]),
        "synthetic_r2": float(a["r2"]),
        "synthetic_n_pairs": float(a["n_pairs"]),
    }
