"""Fourier-Hermite series: density expansion and option pricing.

A density ``f`` on standardized ``z = (x - mu)/sigma`` is expanded in the
probabilists' Hermite basis under the Gaussian weight ``phi``:

    f(z) = phi(z) * sum_n (a_n / n!) He_n(z),

    a_n = E_f[He_n(Z)]   (orthogonality: E_phi[He_m He_n] = n! delta_mn).

For a standardized variable, ``a_0 = 1`` and ``a_1 = a_2 = 0``; higher
coefficients encode skew (a_3), excess kurtosis (a_4) and beyond. The
classic Gram-Charlier/Edgeworth density (``models/edgeworth.py``) is the
order-4 truncation of this series.

Option prices integrate the truncated expansion termwise: with
``xi = (k - mu)/sigma``,

    E[(X - k)^+] = sigma * sum_n a_n (A_n - xi B_n),

    B_n = int_xi^inf phi(z) He_n(z) dz     = phi(xi) He_{n-1}(xi),
    A_n = int_xi^inf z phi(z) He_n(z) dz   = phi(xi)(He_n(xi) + n He_{n-2}(xi)),

using ``z He_n = He_{n+1} + n He_{n-1}``.

References
----------
- Madan & Milne (1994). Contingent claims valued and hedged by pricing
  and investing in a basis. *Math. Finance* 4 — Hermite-basis pricing
  (journal; no arXiv — noted).
- Ackerer, Filipovic & Pulido (2018). The Jacobi stochastic volatility
  model. *Finance Stoch.* — arXiv:1605.07056 (verified; polynomial
  expansions for pricing in bounded state spaces).
- Gram-Charlier / Edgeworth — Charlier (1905), Edgeworth (1905);
  Jarrow & Rudd (1982) option pricing application (journal).
- Escanedes-Filipovic (2023), Fourier-Hermite expansions for option
  pricing — arXiv:2301.08646 (verified; the Fourier-space Hermite
  coefficient route).

Honesty
-------
All benches run on seeded SYNTHETIC densities and smiles generated
in-module. Accuracy numbers validate the series machinery only — never
market evidence. Truncated series can go locally negative; ``repair_density``
clips and renormalizes — the bench reports the raw negativity mass so the
failure mode is visible rather than hidden.

Composition notes
-----------------
- ``models/edgeworth.py``: order-4 Gram-Charlier/Edgeworth special case;
  this module is the arbitrary-order series + pricing.
- ``models/fourier_pricing.py``: COS-method Fourier pricing (different
  basis — cosine on a bounded strip, not Hermite/Gaussian weight).
- ``models/breeden_litzenberger.py`` (wave 24): RND extraction —
  Hermite coefficients of the extracted density are a downstream use.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _check_order(order: int) -> None:
    if order < 3 or order > 12:
        raise ValueError("order must be in [3, 12]")


def he_prob(n: int, z: FloatArray) -> FloatArray:
    """Probabilists' Hermite polynomial He_n evaluated at z.

    Recurrence: He_{n+1} = z He_n - n He_{n-1}.
    """
    if n < 0 or n > 20:
        raise ValueError("n in [0, 20]")
    zz = np.asarray(z, dtype=float)
    if n == 0:
        return np.ones_like(zz)
    if n == 1:
        return zz
    h0 = np.ones_like(zz)
    h1 = zz
    for k in range(1, n):
        h0, h1 = h1, zz * h1 - k * h0
    return h1


def hermite_coefficients(x: FloatArray, order: int = 6) -> FloatArray:
    """Sample estimates of a_n = E[He_n(Z)] for standardized X.

    Returns coefficients a_0..a_order; for a standardized sample
    a_0 ~ 1, a_1 ~ a_2 ~ 0.
    """
    _check_order(order)
    v = np.asarray(x, dtype=float).ravel()
    if v.size < 40 or not np.isfinite(v).all():
        raise ValueError("need >= 40 finite samples")
    mu, sd = float(v.mean()), float(v.std())
    if sd <= 0:
        raise ValueError("degenerate series")
    z = (v - mu) / sd
    a = np.empty(order + 1)
    for n in range(order + 1):
        a[n] = float(np.mean(he_prob(n, z)))
    return a


def hermite_density(z: FloatArray, coeffs: FloatArray, sigma: float = 1.0) -> FloatArray:
    """Evaluate the truncated Hermite-series density at standardized z."""
    if sigma <= 0:
        raise ValueError("sigma > 0")
    zz = np.asarray(z, dtype=float)
    c = np.asarray(coeffs, dtype=float)
    series = np.zeros_like(zz)
    for n in range(c.size):
        series += c[n] * he_prob(n, zz) / math.factorial(n)
    return np.asarray(norm.pdf(zz) * series / sigma)


def repair_density(z: FloatArray, dens: FloatArray) -> tuple[FloatArray, float]:
    """Clip negatives and renormalize; returns (density, negative_mass)."""
    d = np.asarray(dens, dtype=float)
    zg = np.asarray(z, dtype=float)
    if d.shape != zg.shape:
        raise ValueError("shape mismatch")
    neg_mass = float(-d[d < 0].sum())
    clipped = np.clip(d, 0.0, None)
    mass = float(np.trapezoid(clipped, zg))
    if mass <= 0:
        raise ValueError("density has no positive mass")
    return clipped / mass, neg_mass


def hermite_moments(coeffs: FloatArray) -> dict[str, float]:
    """Implied standardized moments from coefficients (up to order 6)."""
    c = np.asarray(coeffs, dtype=float)
    if c.size < 7:
        raise ValueError("need coefficients through order 6")
    # E[Z^n] = sum_k (a_k/k!) * E[He_k(Z) Z^n]
    # For standardized series: E[Z^3] = a_3 (since Z^3 = He_3 + 3He_1)
    # E[Z^4] = 3 + a_4,  E[Z^5] = a_5 + 10 a_3,  E[Z^6] = 15 + 15 a_4 + a_6
    return {
        "skew": float(c[3]),
        "exkurt": float(c[4]),
        "mom5": float(c[5] + 10.0 * c[3]),
        "mom6": float(15.0 + 15.0 * c[4] + c[6]),
    }


def hermite_call_price(coeffs: FloatArray, k: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """E[(X - k)^+] under the Hermite-series density for X.

    ``X = mu + sigma Z`` with Z's density given by the expansion.
    """
    if sigma <= 0:
        raise ValueError("sigma > 0")
    c = np.asarray(coeffs, dtype=float)
    xi = (k - mu) / sigma
    phi_xi = float(norm.pdf(xi))
    bar = float(norm.sf(xi))
    total = phi_xi - xi * bar  # n = 0 term with a_0 = 1 implicit
    for n in range(1, c.size):
        if abs(c[n]) < 1e-15:
            continue
        he_n = float(he_prob(n, np.asarray(xi)))
        he_nm1 = float(he_prob(n - 1, np.asarray(xi)))
        he_nm2 = float(he_prob(n - 2, np.asarray(xi))) if n >= 2 else 0.0
        a_n = phi_xi * (he_n + n * he_nm2)
        b_n = phi_xi * he_nm1
        total += c[n] / math.factorial(n) * (a_n - xi * b_n)
    return float(sigma * total)


def hermite_put_price(coeffs: FloatArray, k: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """E[(k - X)^+] via put-call symmetry on the standardized variable."""
    if sigma <= 0:
        raise ValueError("sigma > 0")
    call = hermite_call_price(coeffs, k, mu, sigma)
    return float(call - (mu - k))


def synth_mixture_samples(n: int, seed: int = 0) -> FloatArray:
    """SYNTHETIC skewed-mixture samples for density-recovery tests."""
    rng = np.random.default_rng(seed)
    pick = rng.random(n) < 0.7
    out = np.where(
        pick,
        rng.normal(0.0, 1.0, n),
        rng.normal(2.0, 1.5, n),
    )
    return np.asarray(out)


def synth_smile_prices(strikes: FloatArray, seed: int = 0) -> FloatArray:
    """SYNTHETIC call prices from a skewed mixture (per-strike expectation)."""
    ks = np.asarray(strikes, dtype=float)
    rng = np.random.default_rng(seed)
    n = 20000
    pick = rng.random(n) < 0.65
    x = np.where(pick, rng.normal(0.0, 1.0, n), rng.normal(1.8, 1.4, n))
    return np.asarray([max(0.0, float(np.mean(np.maximum(x - k, 0.0)))) for k in ks])


def bench_fourier_hermite(seed: int = 20260201) -> dict[str, float]:
    """SYNTHETIC bench for Fourier-Hermite density/pricing. Correctness only."""
    out: dict[str, float] = {}
    # --- density recovery on a skewed mixture --------------------------
    x = synth_mixture_samples(4000, seed=seed)
    for order in (4, 6, 8):
        a = hermite_coefficients(x, order=order)
        zg = np.linspace(-6, 6, 800)
        d = hermite_density(zg, a)
        repaired, neg_mass = repair_density(zg, d)
        out[f"synthetic_neg_mass_o{order}"] = neg_mass
        out[f"synthetic_mass_o{order}"] = float(np.trapezoid(repaired, zg))
        # convergence: coefficient tail shrinkage
        out[f"synthetic_tailcoef_o{order}"] = float(abs(a[-1]) + abs(a[-2]))
    # --- coefficient->moment consistency -------------------------------
    a6 = hermite_coefficients(x, order=6)
    moms = hermite_moments(a6)
    z = (x - x.mean()) / x.std()
    out["synthetic_skew_err"] = float(abs(moms["skew"] - np.mean(z**3)))
    out["synthetic_exkurt_err"] = float(abs(moms["exkurt"] - (np.mean(z**4) - 3.0)))
    # --- pricing vs empirical expectation ------------------------------
    strikes = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    xs = synth_mixture_samples(20000, seed=seed + 1)
    a8 = hermite_coefficients(xs, order=8)
    mu, sd = float(xs.mean()), float(xs.std())
    errs = []
    for k in strikes:
        emp = float(np.mean(np.maximum(xs - k, 0.0)))
        est = hermite_call_price(a8, k, mu=mu, sigma=sd)
        errs.append(abs(est - emp) / max(emp, 1e-6))
    out["synthetic_call_relerr_max"] = float(max(errs))
    out["synthetic_call_relerr_mean"] = float(np.mean(errs))
    # --- determinism ---------------------------------------------------
    a8b = hermite_coefficients(xs, order=8)
    out["synthetic_determinism"] = float(np.allclose(a8, a8b))
    return out
