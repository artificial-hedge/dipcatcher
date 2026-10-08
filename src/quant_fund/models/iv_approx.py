"""Closed-form implied-volatility approximations (SYNTHETIC).

- **Brenner-Subrahmanyam** (1988): for an at-the-money option the price is nearly
  linear in volatility, giving ``sigma ~ sqrt(2 pi / T) * C / S``.
- **Corrado-Miller** (1996): a quadratic correction that stays accurate away from
  the money,

    sigma = sqrt(2 pi/T)/(S + K e^{-rT}) *
            [ (C - X/2) + sqrt( (C - X/2)^2 - X^2/pi ) ],   X = S - K e^{-rT}.

These avoid the iterative root find used by the exact implied vol while
remaining close to it near the money.

References: M. Brenner, M. Subrahmanyam (1988), Financial Analysts Journal;
C. Corrado, T. Miller (1996), Journal of Banking & Finance.  Fail-closed on
non-positive inputs.
"""

from __future__ import annotations

import numpy as np


def brenner_subrahmanyam_iv(call_price: float, s: float, t: float) -> float:
    """At-the-money implied volatility approximation (forward-ATM)."""
    if call_price <= 0.0 or s <= 0.0 or t <= 0.0:
        raise ValueError("call_price, s and t must be positive")
    return float(np.sqrt(2.0 * np.pi / t) * call_price / s)


def corrado_miller_iv(call_price: float, s: float, k: float, t: float, r: float = 0.0) -> float:
    """Corrado-Miller (1996) closed-form implied volatility from a call price."""
    if call_price <= 0.0 or s <= 0.0 or k <= 0.0 or t <= 0.0:
        raise ValueError("prices, strike and maturity must be positive")
    x = s - k * np.exp(-r * t)
    diff = call_price - x / 2.0
    radicand = diff**2 - x**2 / np.pi
    radicand = max(radicand, 0.0)  # guard the deep-ITM/OTM regime
    return float(np.sqrt(2.0 * np.pi / t) / (s + k * np.exp(-r * t)) * (diff + np.sqrt(radicand)))


def _bsm_call(s, k, t, r, sigma):
    from scipy.stats import norm

    d1 = (np.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * np.sqrt(t))
    return s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d1 - sigma * np.sqrt(t))


def _exact_iv(price, s, k, t, r):
    from scipy.optimize import brentq

    return brentq(lambda v: _bsm_call(s, k, t, r, v) - price, 1e-6, 5.0, xtol=1e-10)


def bench_iv_approx(seed: int = 20261231 + 551) -> dict[str, float]:
    """Verify closed-form IV approximations against BSM inversion."""
    rng = np.random.default_rng(seed)
    bs_err = cm_err = 0.0
    n_atm = 0
    for _ in range(40):
        s = 100.0
        t = float(rng.uniform(0.2, 1.5))
        r = float(rng.uniform(0.0, 0.05))
        sig = float(rng.uniform(0.1, 0.6))
        # at/near the money where the approximations claim accuracy
        k = float(s * np.exp(r * t) * rng.uniform(0.95, 1.05))
        price = _bsm_call(s, k, t, r, sig)
        exact = _exact_iv(price, s, k, t, r)
        atm = abs(k - s * np.exp(r * t)) / s < 0.03
        if atm:
            n_atm += 1
            bs_err = max(bs_err, abs(brenner_subrahmanyam_iv(price, s, t) - exact))
        cm_err = max(cm_err, abs(corrado_miller_iv(price, s, k, t, r) - exact))
    if n_atm == 0:
        raise ValueError("no ATM draws — bench uninformative")
    if bs_err > 0.08:
        raise ValueError(f"Brenner-Subrahmanyam off IV oracle: {bs_err:.4f}")
    if cm_err > 0.05:
        raise ValueError(f"Corrado-Miller off IV oracle: {cm_err:.4f}")
    return {
        "synthetic_bs_max_err": bs_err,
        "synthetic_cm_max_err": cm_err,
        "synthetic_n_atm": float(n_atm),
    }
