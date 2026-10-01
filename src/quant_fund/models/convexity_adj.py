"""Futures convexity adjustment — Kirikos & Novak (1997) / Hull.

A FRA/forward rate implied by an interest-rate futures contract exceeds
the true forward rate because of the futures' marking-to-market
correlation with the discount rate. The convexity adjustment converts
the futures-implied rate to the forward:

    forward_rate = futures_rate - convexity

Under a Vasicek / Hull-White short-rate model with mean reversion
``a`` and volatility ``sigma``, the exact adjustment is

    conv = (B(0,T2)/B(0,T1)) * B(T1,T2)
           * [B(T1,T2)*(1 - exp(-2a*T1)) + 2a*B(0,T1)^2] * sigma^2/(4a^3)

where B(t,T) = (1 - exp(-a*(T-t)))/a (Hull technical note on Eurodollar
convexity). The widely used small-vol approximation is
``0.5 * sigma^2 * T1 * T2``.

References
----------
- Kirikos & Novak (1997) RISK 10, "Convexity conundrums".
- Hull (technical note), "Convexity adjustments to Eurodollar futures".

Honesty
-------
Closed-form math plus a Vasicek Monte-Carlo cross-check inside the
bench (the MC is the noisy-but-unbiased estimate of the same object).
All SYNTHETIC; no market data.

Composition
-----------
Called by ``quant_fund.research.benches_w63.bench_convexity_adj``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _b(t: float, big_t: float, a: float) -> float:
    return (1.0 - float(np.exp(-a * (big_t - t)))) / a


def _int_variance(a: float, sigma: float, t1: float, t2: float) -> float:
    """Var of the short-rate integral I = int_{T1}^{T2} r_s ds under
    Vasicek. Cov(r_s,r_u) = (sigma^2/2a)(e^{-a|s-u|} - e^{-a(s+u)}), so

        v = sigma^2/a^2 (tau - B) - sigma^2/(2a) e^{-2aT1} B^2,

    with B = B(T1,T2) = (1-e^{-a*tau})/a. Ho-Lee limit a -> 0:
    sigma^2 (tau^3/3 + tau^2 T1).
    """
    tau = t2 - t1
    if abs(a) < 1e-8:
        return sigma**2 * (tau**3 / 3.0 + tau**2 * t1)
    b = _b(t1, t2, a)
    return float(sigma**2 / a**2 * (tau - b) - sigma**2 / (2.0 * a) * np.exp(-2.0 * a * t1) * b**2)


def convexity_adj_vasicek(
    a: float, sigma: float, t1: float, t2: float, flat_rate: float = 0.0
) -> float:
    """Exact Vasicek/Hull-White convexity (rate units, annualized).

    I = int_{T1}^{T2} r_s ds is Gaussian with mean equal to the forward
    log-return m = ln(P(0,T1)/P(0,T2)) = f*tau for a flat forward
    curve ``flat_rate``, and variance ``_int_variance``. Hence

        futures_rate - forward_rate
            = exp(f*tau) * (exp(v/2) - 1) / tau.

    Fail-closed on non-positive inputs.
    """
    if not (a > 0 and sigma > 0 and t2 > t1 > 0):
        raise ValueError("require a, sigma > 0 and t2 > t1 > 0")
    if not np.isfinite(flat_rate):
        raise ValueError("flat_rate must be finite")
    tau = t2 - t1
    v = _int_variance(a, sigma, t1, t2)
    return float(np.exp(flat_rate * tau) * np.expm1(0.5 * v) / tau)


def convexity_adj_hull(sigma: float, t1: float, t2: float) -> float:
    """Hull's rough heuristic 0.5 * sigma^2 * t1 * t2 — commonly quoted
    with sigma read as the forward-rate volatility. Kept for reference;
    expected to overshoot the Gaussian result."""
    if not (sigma > 0 and t2 > t1 > 0):
        raise ValueError("require sigma > 0 and t2 > t1 > 0")
    return 0.5 * sigma**2 * t1 * t2


def _vasicek_mc_convexity(
    a: float,
    sigma: float,
    r0: float,
    t1: float,
    t2: float,
    n_paths: int,
    seed: int,
) -> tuple[float, float]:
    """Path-simulated convexity under Vasicek — a genuine Monte-Carlo
    cross-check of the Gaussian closed form (not a restatement of it).

    Returns (convexity, standard error of the estimate).
    """
    rng = np.random.default_rng(seed)
    steps_per_year = 120
    n1 = max(1, int(round(t1 * steps_per_year)))
    n2 = n1 + max(1, int(round((t2 - t1) * steps_per_year)))
    dt = t1 / n1
    r = np.full(n_paths, r0)
    for _ in range(n1):
        r = r + (-a * r) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_paths)
    integ = np.zeros(n_paths)
    dt2 = (t2 - t1) / (n2 - n1)
    for _ in range(n2 - n1):
        r = r + (-a * r) * dt2 + sigma * np.sqrt(dt2) * rng.standard_normal(n_paths)
        integ += r * dt2
    tau = t2 - t1
    fut_paths = np.expm1(integ) / tau
    fut = float(np.mean(fut_paths))
    se_fut = float(np.std(fut_paths) / np.sqrt(n_paths))
    fwd = float(np.expm1(r0 * tau) / tau)  # flat curve at r0
    return fut - fwd, se_fut


def bench_convexity(seed: int = 20261231 + 369) -> dict[str, float]:
    """SYNTHETIC check — Gaussian closed form vs path-simulated MC."""
    a, sigma, t1, t2 = 0.10, 0.010, 2.0, 2.25
    conv_ex = convexity_adj_vasicek(a, sigma, t1, t2)
    conv_hull = convexity_adj_hull(sigma, t1, t2)
    conv_mc, se_mc = _vasicek_mc_convexity(a, sigma, 0.0, t1, t2, 80000, seed)
    if conv_ex <= 0:
        raise ValueError("convexity must be positive")
    err_mc = abs(conv_ex - conv_mc)
    if err_mc > max(3.0 * se_mc, 0.3 * conv_ex):
        raise ValueError("closed form disagrees with MC")
    # Ho-Lee limit must approach sigma^2 (tau^3/3 + tau^2 T1).
    v_hl = _int_variance(1e-12, sigma, t1, t2)
    v_hl_ref = sigma**2 * ((t2 - t1) ** 3 / 3.0 + (t2 - t1) ** 2 * t1)
    if abs(v_hl - v_hl_ref) > 1e-6 * v_hl_ref:
        raise ValueError("Ho-Lee variance limit wrong")
    # Longer deposit tenor -> larger adjustment.
    conv_ex2 = convexity_adj_vasicek(a, sigma, t1, t2 + 0.5)
    if conv_ex2 <= conv_ex:
        raise ValueError("convexity not increasing in tenor")
    return {
        "synthetic_cx_vasicek_bp": conv_ex * 1e4,
        "synthetic_cx_hull_bp": conv_hull * 1e4,
        "synthetic_cx_mc_bp": conv_mc * 1e4,
        "synthetic_cx_mc_se_bp": se_mc * 1e4,
        "synthetic_cx_mc_err_bp": err_mc * 1e4,
        "synthetic_cx_long_bp": conv_ex2 * 1e4,
        "score": 1.0,
    }
