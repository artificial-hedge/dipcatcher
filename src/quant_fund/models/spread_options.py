"""Spread / exchange / quanto options — Margrabe (1978), Kirk (1995).

Margrabe prices an exchange option ``max(S2_T - S1_T, 0)`` exactly by
taking S1 as numeraire — the price is Black-76 on the ratio with
volatility sigma = sqrt(s1^2 + s2^2 - 2 rho s1 s2):

    V = S2_0 Phi(d1) - S1_0 Phi(d2)
    d1 = (ln(S2/S1) + sigma^2 T/2) / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T)

Kirk (1995) approximates a spread option max(S2 - S1 - K, 0) on a
nonzero strike by freezing S1 at its forward: treat S1+K as a
lognormal "asset" with adjusted vol s_a = s1 * S1/(S1+K) so the
composite vol is sqrt(s2^2 + s_a^2 - 2 rho s2 s_a) and the formula
reduces to Margrabe on (S2, S1+K).

A quanto (fixed-fx) call pays (S_T - K)^+ in a different currency —
the underlying drifts at r_d - q - rho_q s_S s_FX under the domestic
measure, reducing to Black-76 with drift-adjusted forward.

References
----------
- Margrabe, W. (1978). "The value of an option to exchange one asset
  for another." *Journal of Finance* 33(1).
- Kirk, E. (1995). "Correlation in the energy markets." In *Managing
  Energy Price Risk*, Risk Publications.
- Carmona, R., Durrleman, V. (2003). "Pricing and hedging spread
  options." *SIAM Review* 45(4).

Honesty
-------
SYNTHETIC pricing only; the bench verifies Margrabe against an
independent 2-D Monte Carlo and Kirk against the same MC within its
documented accuracy band.

Composition
-----------
Called by ``quant_fund.research.benches_w65.bench_spread_options``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def margrabe(s1: float, s2: float, sigma1: float, sigma2: float, rho: float, t: float) -> float:
    """Exchange option max(S2_T - S1_T, 0) — exact under GBM."""
    if not (s1 > 0 and s2 > 0 and sigma1 > 0 and sigma2 > 0 and t > 0):
        raise ValueError("spots, vols, horizon must be positive")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho in [-1, 1]")
    sig = np.sqrt(sigma1**2 + sigma2**2 - 2.0 * rho * sigma1 * sigma2)
    if sig <= 0:
        return max(s2 - s1, 0.0)
    sd = sig * np.sqrt(t)
    d1 = (np.log(s2 / s1) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    return float(s2 * norm.cdf(d1) - s1 * norm.cdf(d2))


def kirk_spread(
    s1: float,
    s2: float,
    k: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    t: float,
) -> float:
    """Kirk approximation for max(S2 - S1 - K, 0).

    Freezes X = S1 + K as a lognormal asset with vol scaled by
    S1/(S1+K); the strike-K spread becomes an exchange option.
    """
    if k < 0:
        raise ValueError("use margrabe for K<=0")
    x = s1 + k
    sa = sigma1 * s1 / x
    sig = np.sqrt(sigma2**2 + sa**2 - 2.0 * rho * sigma2 * sa)
    if sig <= 0:
        return max(s2 - x, 0.0)
    sd = sig * np.sqrt(t)
    d1 = (np.log(s2 / x) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    return float(s2 * norm.cdf(d1) - x * norm.cdf(d2))


def quanto_call(
    s0: float,
    k: float,
    r_d: float,
    q: float,
    sigma_s: float,
    sigma_fx: float,
    rho: float,
    t: float,
) -> float:
    """Quanto call: drift r_d - q - rho sigma_s sigma_fx; Black-76 form."""
    if not (s0 > 0 and k > 0 and sigma_s > 0 and t > 0):
        raise ValueError("bad inputs")
    fwd = s0 * np.exp((r_d - q - rho * sigma_s * sigma_fx) * t)
    sd = sigma_s * np.sqrt(t)
    d1 = (np.log(fwd / k) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    return float(np.exp(-r_d * t) * (fwd * norm.cdf(d1) - k * norm.cdf(d2)))


def _mc_spread(
    s1: float,
    s2: float,
    k: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    t: float,
    n: int,
    rng: np.random.Generator,
) -> float:
    z = rng.standard_normal((n, 2))
    c = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    z = z @ c.T
    x1 = s1 * np.exp(-0.5 * sigma1**2 * t + sigma1 * np.sqrt(t) * z[:, 0])
    x2 = s2 * np.exp(-0.5 * sigma2**2 * t + sigma2 * np.sqrt(t) * z[:, 1])
    return float(np.mean(np.maximum(x2 - x1 - k, 0.0)))


def bench_spread_options(seed: int = 20261231 + 381) -> dict[str, float]:
    """SYNTHETIC check — Margrabe/Kirk vs MC, quanto drift sanity."""
    rng = np.random.default_rng(seed)
    s1, s2 = 80.0, 100.0
    sigma1, sigma2, rho, t = 0.25, 0.30, 0.5, 1.0
    v_m = margrabe(s1, s2, sigma1, sigma2, rho, t)
    v_mc = _mc_spread(s1, s2, 0.0, sigma1, sigma2, rho, t, 200000, rng)
    err_m = abs(v_m - v_mc) / v_mc
    if err_m > 0.02:
        raise ValueError("Margrabe deviates from MC")
    k = 12.0
    v_k = kirk_spread(s1, s2, k, sigma1, sigma2, rho, t)
    v_kmc = _mc_spread(s1, s2, k, sigma1, sigma2, rho, t, 200000, rng)
    err_k = abs(v_k - v_kmc) / v_kmc
    if err_k > 0.04:
        raise ValueError("Kirk outside its documented accuracy band")
    q = quanto_call(100.0, 100.0, 0.03, 0.02, 0.2, 0.1, -0.3, 1.0)
    # rho<0 lifts the quanto drift -> price above the rho=0 case.
    q0 = quanto_call(100.0, 100.0, 0.03, 0.02, 0.2, 0.1, 0.0, 1.0)
    if not (q > q0 > 0.0):
        raise ValueError("quanto drift sanity failed")
    return {
        "synthetic_spread_margrabe_err": float(err_m),
        "synthetic_spread_kirk_err": float(err_k),
        "synthetic_spread_margrabe": float(v_m),
        "synthetic_spread_quanto_bp": float((q - q0) * 1e4),
        "synthetic_score": 1.0,
    }
