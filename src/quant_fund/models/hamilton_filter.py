"""Business-cycle decomposition: Hamilton (2018)
regression filter, Hodrick-Prescott, Baxter-King and
Christiano-Fitzgerald band-pass filters.

- Hamilton (2018): regress y_t on a constant + y_{t-h}
  .. y_{t-h-p+1} (h=8 quarters, p=4); residual is the
  stationary cycle. Avoids the spurious-cycle problem
  of HP at the edges.
- HP filter: standard lambda-penalized trend via
  sparse second-difference solve.
- Baxter-King (1999): symmetric approximate band-pass
  FIR (6-32 quarters default).
- Christiano-Fitzgerald (2003): asymmetric/random-walk
  optimal band-pass approximation.

References
----------
- Hamilton (2018) 'Why you should never use the
  Hodrick-Prescott filter' Rev. Econ. Stat. 100(5).
- Hodrick & Prescott (1997) 'Postwar U.S. business
  cycles' JMCB 29(1).
- Baxter & King (1999) 'Measuring business cycles:
  approximate band-pass filters' ReStat 81(4).
- Christiano & Fitzgerald (2003) 'The band pass
  filter' Int. Econ. Rev. 44(2).

Honesty
-------
SYNTHETIC self-check: AR(2) cycle + linear trend; the
Hamilton cycle should recover the known stationary
component within correlation tolerance.

Composition
-----------
Pure numpy. Inputs are a 1-D macro series; outputs
are trend/cycle decompositions per method.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import linalg as spla

FloatArray = NDArray[np.float64]


def _check_series(x: FloatArray, min_n: int = 20) -> FloatArray:
    xa = np.asarray(x, dtype=np.float64).ravel()
    if xa.size < min_n or not np.isfinite(xa).all():
        raise ValueError("series too short or non-finite")
    return xa


def hamilton_filter(x: FloatArray, h: int = 8, p: int = 4) -> tuple[FloatArray, FloatArray]:
    """Hamilton (2018) filter: returns (trend, cycle).

    Cycle_t = y_t - E[y_t | y_{t-h}..y_{t-h-p+1}].
    The first h+p-1 observations get NaN (regression needs
    the full lag set).
    """
    y = _check_series(x, h + p + 10)
    n = y.size
    if h < 1 or p < 1 or n <= h + p:
        raise ValueError("bad h, p for series length")
    X = np.ones((n - h - p + 1, p + 1))
    Y = np.zeros(n - h - p + 1)
    for i in range(n - h - p + 1):
        t = i + h + p - 1
        X[i, 1:] = y[t - h - p + 1 : t - h + 1][::-1]
        Y[i] = y[t]
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    fitted = X @ beta
    cycle = np.full(n, np.nan)
    cycle[h + p - 1 :] = Y - fitted
    trend = np.full(n, np.nan)
    trend[h + p - 1 :] = fitted
    return trend, cycle


def hp_filter(x: FloatArray, lam: float = 1600.0) -> tuple[FloatArray, FloatArray]:
    """Hodrick-Prescott: trend minimizes sum (y-g)^2 + lam*sum(D2g)^2."""
    y = _check_series(x)
    n = y.size
    if lam <= 0:
        raise ValueError("lam positive")
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n)).tocsr()
    A = sparse.eye(n) + lam * (D.T @ D)
    trend = spla.spsolve(A, y)
    return np.asarray(trend), y - np.asarray(trend)


def bk_filter(x: FloatArray, low: int = 6, high: int = 32, k: int = 12) -> FloatArray:
    """Baxter-King symmetric band-pass (cycle only, k leads/lags lost).

    Frequencies passed: periods between `low` and `high`.
    Returns NaN-padded cycle of same length.
    """
    y = _check_series(x, 2 * k + 10)
    if low < 2 or high <= low or k < 1:
        raise ValueError("bad band-pass params")
    j = np.arange(1, k + 1)
    omega_low = 2 * np.pi / high
    omega_high = 2 * np.pi / low
    # ideal band-pass weights
    b = (np.sin(omega_high * j) - np.sin(omega_low * j)) / (np.pi * j)
    b0 = (omega_high - omega_low) / np.pi
    # symmetric filter must sum to zero (remove mean): adjust b0
    adj = (b0 + 2 * b.sum()) / (2 * k + 1)
    w = np.concatenate([b[::-1] - adj, [b0 - adj], b - adj])
    padded = np.convolve(y, w, mode="valid")
    cyc = np.full(y.size, np.nan)
    cyc[k : k + padded.size] = padded
    return cyc


def cf_filter(x: FloatArray, low: int = 6, high: int = 32) -> FloatArray:
    """Christiano-Fitzgerald asymmetric band-pass (random-walk
    optimal weights), NaN-free."""
    y = _check_series(x, 2 * high)
    n = y.size
    if low < 2 or high <= low:
        raise ValueError("bad band")
    omega_low = 2 * np.pi / high
    omega_high = 2 * np.pi / low
    b0 = (omega_high - omega_low) / np.pi
    cyc = np.full(n, np.nan)
    lag0 = np.arange(n)
    for t in range(n):
        # weight on y[i] is the ideal band-pass weight at lag t-i,
        # demeaned over the window (CF asymmetric full-sample form)
        lag = t - lag0
        nz = lag != 0
        w = np.zeros(n)
        w[nz] = (np.sin(omega_high * lag[nz]) - np.sin(omega_low * lag[nz])) / (np.pi * lag[nz])
        w[~nz] = b0
        w -= w.sum() / n
        cyc[t] = float(w @ y)
    return cyc


def bench_hamilton(seed: int = 512) -> dict[str, float]:
    """SYNTHETIC: linear trend + AR(2) business cycle at
    quarterly freq; Hamilton cycle should correlate with
    the true cycle; HP should track trend."""
    rng = np.random.default_rng(seed)
    n = 200
    trend = 0.5 * np.arange(n)
    # AR(2) cycle with business-cycle periodicity ~ 24-32 quarters
    rho1, rho2 = 1.35, -0.55
    eps = rng.normal(0, 1.0, n)
    cyc_true = np.zeros(n)
    for t in range(2, n):
        cyc_true[t] = rho1 * cyc_true[t - 1] + rho2 * cyc_true[t - 2] + eps[t]
    y = trend + cyc_true + 100.0
    _, cyc_h = hamilton_filter(y, h=8, p=4)
    tr_hp, cyc_hp = hp_filter(y, lam=1600)
    cyc_bk = bk_filter(y)
    valid = ~np.isnan(cyc_h)
    corr_h = float(np.corrcoef(cyc_h[valid], cyc_true[valid])[0, 1])
    corr_hp = float(np.corrcoef(cyc_hp, cyc_true)[0, 1])
    vb = ~np.isnan(cyc_bk)
    corr_bk = float(np.corrcoef(cyc_bk[vb], cyc_true[vb])[0, 1])
    if corr_h < 0.7 or corr_hp < 0.7:
        raise ValueError("cycle recovery poor")
    return {
        "synthetic_corr_hamilton": corr_h,
        "synthetic_corr_hp": corr_hp,
        "synthetic_corr_bk": corr_bk,
        "synthetic_cycle_sd_h": float(np.nanstd(cyc_h)),
        "synthetic_cycle_sd_true": float(np.std(cyc_true)),
        "synthetic_cycle_rmse_hp": float(np.sqrt(((cyc_hp - cyc_true) ** 2).mean())),
    }
