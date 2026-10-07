"""CBOE-style variance-swap replication from an OTM option strip (SYNTHETIC).

Carr & Madan (1998) / Demeterfi, Derman, Kamal & Zou (1999) show the fair
strike of a variance swap is replicated by a static strip of out-of-the-money
options weighted by 1/K^2:

    sigma_rep^2 = (2/T) e^{rT} sum_i [ dK_i / K_i^2 ] Q(K_i)
                  - (1/T) ( F / K_0 - 1 )^2

where Q(K) is the midprice of the OTM option at strike K, F is the forward
implied by put-call parity, and K_0 is the first strike at or below F. The
"corridor" variant (Andersen & Bondarenko 2007) truncates the strip to
[K_lo, K_hi] which targets the corridor variance swap.

Honesty: prices must come from an honest model or real data; the bench prices
a Black-Scholes surface so the replicated variance can be verified against the
planted true variance. Fail-closed on bad grids, negative prices, or a strip
that does not bracket the forward.

References: Demeterfi et al. (1999) GS "More Than You Ever Wanted to Know
About Volatility Swaps"; Carr & Wu (2009) "Variance Risk Premiums";
Andersen & Bondarenko (2007) "Construction of smooth volatility surface".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def implied_forward(
    strikes: FloatArray, calls: FloatArray, puts: FloatArray, r: float, t: float
) -> tuple[float, float]:
    """(F, K0): forward from put-call parity at the strike minimising
    |C - P|, plus the first strike at or below F."""
    k = np.asarray(strikes, dtype=float).ravel()
    c = np.asarray(calls, dtype=float).ravel()
    p = np.asarray(puts, dtype=float).ravel()
    if not (k.size == c.size == p.size) or k.size < 3:
        raise ValueError("strikes/calls/puts must match and have >= 3 points")
    if np.any(np.diff(k) <= 0):
        raise ValueError("strikes must be strictly increasing")
    if np.any(~np.isfinite(c)) or np.any(~np.isfinite(p)):
        raise ValueError("non-finite option prices")
    j = int(np.argmin(np.abs(c - p)))
    fwd = float(k[j] + np.exp(r * t) * (c[j] - p[j]))
    below = k[k <= fwd]
    if below.size == 0:
        raise ValueError("strip does not bracket the forward")
    return fwd, float(below[-1])


def variance_strike(
    strikes: FloatArray,
    otm_prices: FloatArray,
    fwd: float,
    k0: float,
    r: float,
    t: float,
) -> float:
    """Replicated annualized variance strike from the OTM strip."""
    k = np.asarray(strikes, dtype=float).ravel()
    q = np.asarray(otm_prices, dtype=float).ravel()
    if k.size != q.size or k.size < 3:
        raise ValueError("strike/price mismatch or too few strikes")
    if np.any(q < 0):
        raise ValueError("negative option prices")
    if not (0.0 < t <= 10.0):
        raise ValueError("expiry out of range")
    dk = np.empty(k.size)
    dk[1:-1] = 0.5 * (k[2:] - k[:-2])
    dk[0] = k[1] - k[0]
    dk[-1] = k[-1] - k[-2]
    contrib = dk / (k * k) * q
    var = (2.0 / t) * np.exp(r * t) * contrib.sum() - (1.0 / t) * (fwd / k0 - 1.0) ** 2
    return float(var)


def otm_strip(strikes: FloatArray, calls: FloatArray, puts: FloatArray, k0: float) -> FloatArray:
    """Select the OTM leg: puts below K0, calls above, min(C,P) at K0."""
    k = np.asarray(strikes, dtype=float).ravel()
    c = np.asarray(calls, dtype=float).ravel()
    p = np.asarray(puts, dtype=float).ravel()
    q = np.where(k < k0, p, np.where(k > k0, c, np.minimum(c, p)))
    return np.asarray(q, dtype=np.float64)


def corridor_strike(
    strikes: FloatArray,
    calls: FloatArray,
    puts: FloatArray,
    k_lo: float,
    k_hi: float,
    r: float,
    t: float,
) -> float:
    """Corridor variance swap strike on [K_lo, K_hi] (options outside the
    corridor enter at zero — the corridor contracts are worthless there).

    sigma_cor^2 = (2/T) e^{rT} int_{K_lo}^{K_hi} Q(K)/K^2 dK

    which does not subtract the forward-term adjustment (the corridor
    swap's payoff is capped at the corridor edges instead).
    """
    k = np.asarray(strikes, dtype=float).ravel()
    c = np.asarray(calls, dtype=float).ravel()
    p = np.asarray(puts, dtype=float).ravel()
    if not (k_lo < k_hi):
        raise ValueError("empty corridor")
    if not (k[0] <= k_lo and k[-1] >= k_hi):
        raise ValueError("strip must span the corridor")
    if not (0.0 < t <= 10.0):
        raise ValueError("expiry out of range")
    m = (k >= k_lo) & (k <= k_hi)
    if m.sum() < 3:
        raise ValueError("corridor covers fewer than 3 strikes")
    kc = k[m]
    # inside the corridor, the OTM convention flips at the corridor's
    # forward crossing; use puts below the strip median, calls above.
    k_med = 0.5 * (k_lo + k_hi)
    q = np.where(kc < k_med, p[m], c[m])
    dk = np.empty(kc.size)
    dk[1:-1] = 0.5 * (kc[2:] - kc[:-2])
    dk[0] = kc[1] - kc[0]
    dk[-1] = kc[-1] - kc[-2]
    return float((2.0 / t) * np.exp(r * t) * np.sum(dk * q / (kc * kc)))


def bench_vix_replication(seed: int = 20261231 + 396) -> dict[str, float]:
    """SYNTHETIC check — replicated variance matches planted BS vol."""
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    s0, r, t, sig = 100.0, 0.03, 30.0 / 365.0, 0.22
    fwd = s0 * np.exp(r * t)
    strikes = np.linspace(70.0, 140.0, 141)
    d1 = (np.log(fwd / strikes) + 0.5 * sig * sig * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)
    disc = np.exp(-r * t)
    calls = disc * (fwd * norm.cdf(d1) - strikes * norm.cdf(d2))
    puts = calls - disc * (fwd - strikes)
    # small multiplicative noise on mids (keeps prices positive)
    calls = calls * (1.0 + 1e-4 * rng.standard_normal(calls.size))
    puts = puts * (1.0 + 1e-4 * rng.standard_normal(puts.size))

    f_hat, k0 = implied_forward(strikes, calls, puts, r, t)
    if abs(f_hat - fwd) > 0.5:
        raise ValueError("forward recovery off")
    q = otm_strip(strikes, calls, puts, k0)
    var_rep = variance_strike(strikes, q, f_hat, k0, r, t)
    sig_rep = float(np.sqrt(var_rep))
    sig_err = abs(sig_rep - sig)
    if sig_err > 0.02:
        raise ValueError(f"replicated vol {sig_rep} vs planted {sig}")
    cor = corridor_strike(strikes, calls, puts, 85.0, 115.0, r, t)
    if not (0.0 < cor < var_rep * 1.3):
        raise ValueError("corridor strike implausible")
    return {
        "synthetic_vix_sig_rep": sig_rep,
        "synthetic_vix_sig_err": sig_err,
        "synthetic_vix_fwd_err": abs(f_hat - fwd),
        "synthetic_vix_corridor": float(np.sqrt(cor)),
        "synthetic_score": 1.0,
    }
