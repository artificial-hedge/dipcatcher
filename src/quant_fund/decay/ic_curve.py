"""Horizon-IC curves: predictive content as a function of holding lag.

Builds forward-return matrices from a period-return panel, then scores a
signal's rank correlation at each lag k = 1..max_lag. The curve's AUC, peak
lag, and decay classification tell a researcher where the signal stops
carrying information — the input to position sizing and combination.

Honesty: the curve is a descriptive diagnostic on the supplied panel, built
from proper rank correlations; it is not an expected-performance profile.

References:
- Grinold, R., Kahn, R. (2000). *Active Portfolio Management* — IC decay.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*,
  ch. 3–5 — horizon-aligned labels and decay.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.decay.ic_series import spearman_ic

FloatArray = NDArray[np.float64]


def forward_returns_matrix(period_returns: FloatArray, max_lag: int) -> FloatArray:
    """Forward compounded return matrix R[t, k] = Π_{j=1..k}(1+r_{t+j}) − 1.

    Row t holds the k-period forward return from t for k = 1..max_lag; the
    window covers periods t+1..t+k (period t itself is not included). Rows
    where any constituent return is missing are NaN (no peeking).
    """
    r = np.asarray(period_returns, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError("period_returns must be one-dimensional")
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")
    n = len(r)
    out = np.full((n - max_lag, max_lag), np.nan, dtype=np.float64)
    log1p = np.log1p(r)
    cums = np.concatenate(([0.0], np.cumsum(log1p)))
    # cums[m] = sum of log1p up to index m-1; the forward window for lag k
    # is ret[t+1..t+k] → cums[t+1+k] - cums[t+1]
    for t in range(n - max_lag):
        out[t, :] = np.expm1(cums[t + 2 : t + 2 + max_lag] - cums[t + 1])
    return out


def compound_forward_returns(period_returns: FloatArray, max_lag: int) -> FloatArray:
    """Alias kept for call-site readability (same construction as above)."""
    return forward_returns_matrix(period_returns, max_lag)


def ic_curve(pred: FloatArray, fwd: FloatArray) -> FloatArray:
    """Rank IC of the signal at each forward lag.

    ``pred`` is (T, N) and ``fwd`` is (T, max_lag, N): the IC at lag k uses
    fwd[:, k, :].
    """
    pred = np.asarray(pred, dtype=np.float64)
    fwd = np.asarray(fwd, dtype=np.float64)
    if fwd.ndim != 3:
        raise ValueError("fwd must be (T, max_lag, N)")
    max_lag = fwd.shape[1]
    if pred.shape[0] != fwd.shape[0] or pred.shape[1] != fwd.shape[2]:
        raise ValueError("pred must be (T, N) with T, N matching fwd")
    out = np.empty(max_lag, dtype=np.float64)
    for k in range(max_lag):
        out[k] = float(np.nanmean(spearman_ic(pred, fwd[:, k, :])))
    return out


def lag_ic(pred_t: FloatArray, ret_t_plus_k: FloatArray) -> float:
    """Single-lag rank IC between signal at t and return at t + k."""
    pred_t = np.asarray(pred_t, dtype=np.float64)
    ret_t_plus_k = np.asarray(ret_t_plus_k, dtype=np.float64)
    if pred_t.shape != ret_t_plus_k.shape:
        raise ValueError("pred and return panels must have equal shapes")
    return float(np.nanmean(spearman_ic(pred_t.reshape(1, -1), ret_t_plus_k.reshape(1, -1))))


def curve_auc(curve: FloatArray) -> float:
    """Area under the |IC| curve, normalised to [0, 1] per unit lag."""
    x = np.asarray(curve, dtype=np.float64)
    x = np.abs(x)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("curve must have at least two lags")
    auc = float(np.trapezoid(x, dx=1.0))
    return float(auc / (len(x) - 1))


def peak_lag(curve: FloatArray) -> int:
    """1-based lag of maximum |IC|."""
    x = np.abs(np.asarray(curve, dtype=np.float64))
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("curve must be non-empty")
    return int(np.nanargmax(x)) + 1
