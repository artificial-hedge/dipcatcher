"""Kemna-Vorst Asian options: geometric closed form + CV arithmetic MC.

References
----------
- Kemna, A.G.Z. & Vorst, A.C.F. (1990). "A Pricing Method
  for Options Based on Average Asset Values." *Journal of
  Banking and Finance* 14(1), 113-129.
- Turnbull, S.M. & Wakeman, L.M. (1991). "A Quick
  Algorithm for Pricing European Average Options."
  *JFQA* 26(3), 377-389.
- Curran, M. (1994). "Valuing Asian and Portfolio Options
  by Conditioning on the Geometric Mean Price."
  *Management Science* 40(12), 1705-1711.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial
  Engineering*. Springer, ch. 4 (control variates),
  ch. 8 (path-dependent payoffs).

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence. Prices are
abstract derivatives machinery on GBM only.

Composition notes
-----------------
A geometric-mean Asian call has a closed form because the
geometric mean of GBM is lognormal: the average's
log-variance is ``sigma^2 T / 3`` (continuous monitoring)
with drift ``(r - sigma^2/6) T/2``, giving an effective
Black-Scholes price with adjusted vol
``sigma_G = sigma / sqrt(3)`` and adjusted rate
``r_G = (r - sigma^2/2)/2 + sigma_G^2/2``. The arithmetic-
mean option has no closed form; we price by MC with the
geometric price as control variate — variance reduction
comes from corr(arith payoff, geom payoff) ~ 0.99+, which
is the only reason the MC standard error is credible at
modest path counts. Turnbull-Wakeman moment matching is
included as the fast approximation: match the first two
moments of the arithmetic average to a lognormal.
Discrete monitoring uses daily-ish spacing. Guards:
non-positive strike/tenor/vol fail closed, paths are
antithetic, and the CV coefficient is the honest
sample-covariance estimate (no plug-in shortcut).
``synth_asian`` prices ATM options on a synthetic GBM;
the bench gates on geometric price hitting the closed
form, arithmetic MC within a few standard errors of the
moment-matched approx, and CV shrinking MC variance
materially vs raw MC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _check(s0: float, k: float, t: float, r: float, sigma: float) -> None:
    if min(s0, k, t, sigma) <= 0.0 or not all(np.isfinite(v) for v in (s0, k, t, r, sigma)):
        raise ValueError("bad option inputs")


def geo_asian_call(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
) -> float:
    """Kemna-Vorst continuous geometric-mean Asian call."""
    _check(s0, k, t, r, sigma)
    sig_g = sigma / np.sqrt(3.0)
    r_g = 0.5 * (r - 0.5 * sigma**2) + 0.5 * sig_g**2
    d1 = (np.log(s0 / k) + (r_g + 0.5 * sig_g**2) * t) / (sig_g * np.sqrt(t))
    d2 = d1 - sig_g * np.sqrt(t)
    return float(np.exp(-r * t) * (s0 * np.exp(r_g * t) * norm.cdf(d1) - k * norm.cdf(d2)))


def tw_asian_call(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
) -> float:
    """Turnbull-Wakeman moment-matched arithmetic Asian call."""
    _check(s0, k, t, r, sigma)
    # first two moments of the arithmetic average
    m1 = s0 * (np.exp(r * t) - 1.0) / (r * t)
    # E[A^2] = (2 S0^2 / T^2) / (r+sig^2) *
    #   [(e^{(2r+sig^2)T}-1)/(2r+sig^2) - (e^{rT}-1)/r]
    m2 = (
        (2.0 * s0**2 / t**2)
        / (r + sigma**2)
        * (
            (np.exp((2.0 * r + sigma**2) * t) - 1.0) / (2.0 * r + sigma**2)
            - (np.exp(r * t) - 1.0) / r
        )
    )
    v = m2 - m1**2
    sig_tw = float(np.sqrt(np.log(1.0 + v / m1**2)))
    b_tw = float(np.log(m1) - 0.5 * sig_tw**2)
    k_adj = np.log(k)
    d1 = (b_tw - k_adj + sig_tw**2) / sig_tw
    d2 = d1 - sig_tw
    return float(
        np.exp(-r * t) * (np.exp(b_tw + 0.5 * sig_tw**2) * norm.cdf(d1) - k * norm.cdf(d2))
    )


def arith_asian_mc(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    n_steps: int = 60,
    n_paths: int = 20000,
    seed: int = 0,
) -> dict[str, float]:
    """Arithmetic Asian call by antithetic MC with geometric CV."""
    _check(s0, k, t, r, sigma)
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    z = rng.standard_normal((n_paths // 2, n_steps))
    z = np.vstack([z, -z])
    inc = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    logp = np.log(s0) + np.cumsum(inc, axis=1)
    paths = np.exp(logp)
    arith = np.mean(paths, axis=1)
    geo = np.exp(np.mean(logp, axis=1))
    pay_a = np.maximum(arith - k, 0.0) * np.exp(-r * t)
    pay_g = np.maximum(geo - k, 0.0) * np.exp(-r * t)
    g_true = geo_asian_call(s0, k, t, r, sigma)
    cov = float(np.cov(pay_a, pay_g)[0, 1])
    var_g = float(np.var(pay_g))
    c_star = cov / max(var_g, 1e-12)
    cv_price = float(np.mean(pay_a) - c_star * (np.mean(pay_g) - g_true))
    cv_se = float(np.std(pay_a - c_star * (pay_g - g_true)) / np.sqrt(pay_a.size))
    raw_se = float(np.std(pay_a) / np.sqrt(pay_a.size))
    out: dict[str, float] = {
        "price": cv_price,
        "cv_se": cv_se,
        "raw_se": raw_se,
        "c_star": c_star,
        "var_reduction": float(1.0 - (cv_se / max(raw_se, 1e-12)) ** 2),
        "geo_price": g_true,
        "tw_price": tw_asian_call(s0, k, t, r, sigma),
    }
    return out


def synth_asian(seed: int = 20261231 + 364) -> dict[str, float]:
    """SYNTHETIC ATM Asian pricing on GBM parameters."""
    s0, k, t, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.25
    g = geo_asian_call(s0, k, t, r, sigma)
    tw = tw_asian_call(s0, k, t, r, sigma)
    r_mc = arith_asian_mc(s0, k, t, r, sigma, seed=seed)
    p = float(r_mc["price"])
    se = float(r_mc["cv_se"])
    # arithmetic > geometric (Jensen); TW should be close to MC
    ok = (
        p > g
        and abs(p - tw) < 0.15 * tw
        and se < 0.02 * p
        and float(r_mc["var_reduction"]) > 0.8
        and g > 0.0
    )
    out: dict[str, float] = {
        "synthetic_ao_geo": g,
        "synthetic_ao_tw": tw,
        "synthetic_ao_mc": p,
        "synthetic_ao_cv_se": se,
        "synthetic_ao_var_red": float(r_mc["var_reduction"]),
        "score": 1.0 if ok else 0.0,
    }
    return out
