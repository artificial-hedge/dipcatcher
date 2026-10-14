"""Realized volatility estimators and jump tests on bar/price series.

Implements the standard high-frequency econometrics toolkit on a grid of
log-prices:

- ``realized_variance`` — the quadratic-variation proxy RV = Σ r²;
- ``bipower_variation`` — BNS (2004) BV = (π/2) Σ |r_t| |r_{t−1}|, robust
  to jumps;
- ``minrv`` / ``medianrv`` — Andersen-Dobrev-Schaumburg (2012) nearest-
  neighbour estimators, robust to jumps and to zero returns in a grid;
- ``realized_kernel`` — Barndorff-Nielsen et al. (2008) Tukey–Hanning flat-
  top kernel with bandwidth selection h = c · n^{4/5};
- ``jump_tests`` — the BNS ratio test and the Huang–Tauchen (2005) z-test
  for the null of no jump component in QV.

Honesty: these are estimators and test statistics on the supplied series.
Jump detection on synthetic data is a correctness fixture only.

References:
- Barndorff-Nielsen, O. E., Shephard, N. (2004). Power and bipower
  variation with stochastic volatility and jumps — BV and the ratio test.
- Andersen, T. G., Dobrev, D., Schaumburg, E. (2012). Jump-robust
  volatility estimation using nearest neighbor truncation — minRV/medianRV.
- Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., Shephard, N. (2008).
  Designing realised kernels to measure ex-post variation — RK.
- Huang, X., Tauchen, G. (2005). The relative contribution of jumps to
  total price variance — the z-test.

Composition: pure numpy; deterministic; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _log_returns(prices: FloatArray) -> FloatArray:
    prices = np.asarray(prices, dtype=np.float64)
    if prices.ndim != 1 or len(prices) < 3:
        raise ValueError("prices must be a one-dimensional array with >= 3 points")
    if np.any(prices <= 0):
        raise ValueError("prices must be positive for log-returns")
    return np.diff(np.log(prices))


def realized_variance(prices: FloatArray) -> float:
    """RV = Σ r² — the sum of squared log-returns (quadratic variation proxy)."""
    r = _log_returns(prices)
    return float(np.sum(r * r))


def bipower_variation(prices: FloatArray) -> float:
    """BV = (π/2) Σ |r_t| |r_{t−1}| — jump-robust integrated variance."""
    r = _log_returns(prices)
    return float((np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))


def minrv(prices: FloatArray) -> float:
    """minRV = (π/(π−2)) · (n/(n−1)) Σ min(r_t², r_{t−1}²) — jump-robust."""
    r = _log_returns(prices)
    n = len(r)
    if n < 2:
        raise ValueError("need at least 2 returns")
    m = np.minimum(r[1:] ** 2, r[:-1] ** 2)
    return float((np.pi / (np.pi - 2.0)) * (n / (n - 1.0)) * np.sum(m))


def medianrv(prices: FloatArray) -> float:
    """medianRV via nearest-neighbour median truncation (jump-robust).

    medianRV = β · (n/(n−2)) Σ med(|r_{t−1}|, |r_t|, |r_{t+1}|)² with the
    Andersen–Dobrev–Schaumburg (2012) constant β = 1.5106, chosen so the
    estimator is asymptotically unbiased for integrated variance under
    i.i.d. Gaussian returns (E[median of 3 |N(0,1)| draws]² = 1/β).
    """
    r = _log_returns(prices)
    n = len(r)
    if n < 3:
        raise ValueError("need at least 3 returns")
    beta = 1.5106
    med = np.median(np.stack([np.abs(r[:-2]), np.abs(r[1:-1]), np.abs(r[2:])]), axis=0)
    return float(beta * (n / (n - 2.0)) * np.sum(med * med))


def _tukey_hanning_weights(h: int) -> FloatArray:
    """Tukey–Hanning weights k(j/h) = sin²(π/2 · (1 − j/h)₊) for j = 1..h."""
    lags = np.arange(1, h + 1, dtype=np.float64)
    x = np.clip(lags / h, 0.0, 1.0)
    return np.sin(0.5 * np.pi * (1.0 - x)) ** 2


# Direct inner products stay cheaper while only a few lags are required.
# Past this, a zero-padded FFT autocorrelation is O(n log n). The default
# bandwidth is ``h ~ n^{4/5}``, which is that regime.
_DIRECT_LAG_LIMIT = 64


def _direct_autocovariances(series: FloatArray, max_lag: int) -> FloatArray:
    out = np.empty(max_lag + 1, dtype=np.float64)
    out[0] = np.dot(series, series)
    for lag in range(1, max_lag + 1):
        out[lag] = np.dot(series[lag:], series[:-lag])
    return out


def _fft_autocovariances(series: FloatArray, max_lag: int) -> FloatArray:
    """Linear (not circular) autocorrelation via a pad of length at least ``2n-1``."""
    n = int(series.size)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(series, size)
    corr = np.fft.irfft(spectrum * np.conjugate(spectrum), size)
    return np.asarray(np.real(corr[: max_lag + 1]), dtype=np.float64)


def _autocovariances(returns: FloatArray, max_lag: int) -> FloatArray:
    """Lag 0..``max_lag`` realized autocovariances, ``gamma_j = sum_t r_t r_{t-j}``."""
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    series = np.ascontiguousarray(returns, dtype=np.float64)
    if int(series.size) <= max_lag:
        raise ValueError("max_lag must be shorter than the return series")
    if max_lag <= _DIRECT_LAG_LIMIT:
        return _direct_autocovariances(series, max_lag)
    return _fft_autocovariances(series, max_lag)


def realized_kernel(prices: FloatArray, bandwidth: int | None = None, c: float = 3.0) -> float:
    """Realized kernel with Tukey–Hanning weights, bandwidth h = c·n^{4/5}.

    RK = γ₀ + 2 Σ_{j=1}^{h} k(j/h) γ_j where γ_j = Σ_t r_t r_{t−j} is the
    j-th realized autocovariance. Short bandwidths use direct inner
    products. The default bandwidth ``h ~ n^{4/5}`` uses a padded FFT,
    which is the same sum up to roundoff well inside the estimator's
    sampling error.
    """
    returns = _log_returns(prices)
    n = len(returns)
    if bandwidth is None:
        bandwidth = max(1, int(c * n**0.8))
    h = min(int(bandwidth), n - 1)
    # A non-positive bandwidth has no lags. The kernel is then γ₀, which is
    # the realized variance of the same log-returns.
    if h < 1:
        return float(np.dot(returns, returns))
    gammas = _autocovariances(returns, h)
    return float(gammas[0] + 2.0 * np.dot(_tukey_hanning_weights(h), gammas[1:]))


def _gaussian_bootstrap(n: int, var_hat: float, n_boot: int, seed: int) -> FloatArray:
    """``n_boot`` i.i.d. Gaussian paths, one ``standard_normal(n)`` draw each.

    Drawing a path at a time keeps the Generator stream identical to the
    scalar bootstrap. The quadratic variation of those paths is then one
    array reduction.
    """
    rng = np.random.default_rng(seed)
    draws = np.empty((int(n_boot), int(n)), dtype=np.float64)
    scale = math.sqrt(var_hat) if var_hat > 0.0 else 0.0
    for i in range(int(n_boot)):
        draws[i] = rng.standard_normal(n)
    draws *= scale
    return draws


def _bootstrap_quadratic(draws: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Per-path realized variance and the bipower sum ``Σ |r_t| |r_{t-1}|``."""
    rv_b = np.sum(draws * draws, axis=1)
    abs_r = np.abs(draws)
    bipower_sum = np.sum(abs_r[:, 1:] * abs_r[:, :-1], axis=1)
    return rv_b, bipower_sum


def bns_ratio_test(
    prices: FloatArray,
    *,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """BNS jump test with a parametric-bootstrap p-value.

    Statistic: RV/BV (→ 1 under no jumps, > 1 with jumps). The null
    distribution is simulated by parametric bootstrap: i.i.d. Gaussian
    returns with variance matched to RV/n (no-jump world). One-sided
    p-value: fraction of bootstrap ratios ≥ observed.
    """
    returns = _log_returns(prices)
    rv = realized_variance(prices)
    bv = bipower_variation(prices)
    if bv <= 0:
        return {"ratio": float("inf"), "p": 0.0}
    n = len(returns)
    ratio = rv / bv
    draws = _gaussian_bootstrap(n, rv / n, n_boot, seed)
    rv_b, bipower_sum = _bootstrap_quadratic(draws)
    bv_b = (np.pi / 2.0) * bipower_sum
    count = int(np.count_nonzero((bv_b > 0.0) & (rv_b >= ratio * bv_b)))
    return {"ratio": float(ratio), "p": float((count + 1) / (n_boot + 1))}


def huang_tauchen_z(
    prices: FloatArray,
    *,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Huang–Tauchen (2005) jump test with parametric-bootstrap p-value.

    Statistic: RV − BV (positive when jumps contribute to QV). The null
    (no jumps) is simulated with i.i.d. Gaussian returns variance-matched to
    RV/n; one-sided p-value from the bootstrap distribution.
    """
    returns = _log_returns(prices)
    rv = realized_variance(prices)
    bv = bipower_variation(prices)
    n = len(returns)
    stat = rv - bv
    draws = _gaussian_bootstrap(n, rv / n, n_boot, seed)
    rv_b, bipower_sum = _bootstrap_quadratic(draws)
    stat_b = rv_b - (np.pi / 2.0) * bipower_sum
    count = int(np.count_nonzero(stat_b >= stat))
    return {"stat": float(stat), "p": float((count + 1) / (n_boot + 1))}


def integrated_jump_component(prices: FloatArray) -> float:
    """max(RV − BV, 0) — the estimated jump contribution to QV."""
    return max(realized_variance(prices) - bipower_variation(prices), 0.0)
