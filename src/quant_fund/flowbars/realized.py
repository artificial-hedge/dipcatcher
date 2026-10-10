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


def _tukey_hanning_weight(j: int, h: int) -> float:
    """Tukey–Hanning weight k(j/h) = sin²(π/2 · (1 − j/h)₊)."""
    x = min(max(j / h, 0.0), 1.0)
    return float(np.sin(0.5 * np.pi * (1.0 - x)) ** 2)


def realized_kernel(prices: FloatArray, bandwidth: int | None = None, c: float = 3.0) -> float:
    """Realized kernel with Tukey–Hanning weights, bandwidth h = c·n^{4/5}.

    RK = γ₀ + 2 Σ_{j=1}^{n−1} k(j/h) γ_j where γ_j = Σ_t r_t r_{t−j} is the
    j-th realized autocovariance.
    """
    r = _log_returns(prices)
    n = len(r)
    if bandwidth is None:
        bandwidth = max(1, int(c * n**0.8))
    h = min(bandwidth, n - 1)
    gamma0 = float(np.sum(r * r))
    total = gamma0
    for j in range(1, h + 1):
        gamma_j = float(np.sum(r[j:] * r[:-j]))
        total += 2.0 * _tukey_hanning_weight(j, h) * gamma_j
    return float(total)


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
    r = _log_returns(prices)
    rv = realized_variance(prices)
    bv = bipower_variation(prices)
    if bv <= 0:
        return {"ratio": float("inf"), "p": 0.0}
    n = len(r)
    ratio = rv / bv
    rng = np.random.default_rng(seed)
    var_hat = rv / n
    count = 0
    for _ in range(n_boot):
        rb = rng.standard_normal(n) * np.sqrt(var_hat)
        rv_b = float(np.sum(rb * rb))
        bv_b = float((np.pi / 2.0) * np.sum(np.abs(rb[1:]) * np.abs(rb[:-1])))
        if bv_b > 0 and rv_b / bv_b >= ratio:
            count += 1
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
    r = _log_returns(prices)
    rv = realized_variance(prices)
    bv = bipower_variation(prices)
    n = len(r)
    stat = rv - bv
    rng = np.random.default_rng(seed)
    var_hat = rv / n
    count = 0
    for _ in range(n_boot):
        rb = rng.standard_normal(n) * np.sqrt(var_hat)
        stat_b = float(np.sum(rb * rb) - (np.pi / 2.0) * np.sum(np.abs(rb[1:]) * np.abs(rb[:-1])))
        if stat_b >= stat:
            count += 1
    return {"stat": float(stat), "p": float((count + 1) / (n_boot + 1))}


def integrated_jump_component(prices: FloatArray) -> float:
    """max(RV − BV, 0) — the estimated jump contribution to QV."""
    return max(realized_variance(prices) - bipower_variation(prices), 0.0)
