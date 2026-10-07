"""Bates (1996) stochastic-volatility + jump-diffusion pricing.

References
----------
- Bates, D.S. (1996). "Jumps and Stochastic Volatility:
  Exchange Rate Processes Implicit in Deutsche Mark Options."
  *Review of Financial Studies* 9(1), 69-107.
- Merton, R.C. (1976). "Option Pricing when Underlying Stock
  Returns are Discontinuous." *Journal of Financial Economics*
  3(1-2), 125-144.
- Heston, S.L. (1993). "A Closed-Form Solution for Options
  with Stochastic Volatility." *Review of Financial Studies*
  6(2), 327-343.
- Kahl, C. & Jaeckel, P. (2005). "Not-so-complex Logarithms in
  the Heston Model." *Wilmott Magazine*, 94-103.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
The Bates CF is Heston's affine CF with an independent Merton
lognormal-jump factor: ``phi(u) = phi_H(u) * exp(lam*T*(m(u)
- i*u*kappa_j))`` where ``m(u) = exp(i*u*muj - u^2*sigj^2/2) -
1`` and the compensator ``kappa_j = exp(muj + sigj^2/2) - 1``
keeps the stock a martingale. We use the "little trap"
formulation (Kahl-Jaeckel g-switching-free branch with the
standard ``d = sqrt((rho*sigma*i*u - kappa)^2 + sigma^2*(i*u +
u^2))`` Riccati coefficients) and Gauss-Legendre quadrature on
the two risk-neutral probabilities — the Heston-plus-jump CF
falls out analytically only when the jump factor is appended
*after* the Heston Riccati solve, not folded into the drift.
``synth_bates`` prices an OTM put/call grid; the bench gates on
jumps raising short-maturity put skew (fat left tail) while
holding the martingale property (forward parity at the CF
level, |phi(-i)| check).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def bates_cf(
    u: complex | ComplexArray,
    s0: float,
    t: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    lam: float,
    muj: float,
    sigj: float,
) -> complex | ComplexArray:
    """Bates characteristic function phi_T(u) of log S_T."""
    if s0 <= 0 or t <= 0 or v0 < 0 or sigma <= 0 or lam < 0 or sigj < 0:
        raise ValueError("bad parameters")
    uu = np.asarray(u, dtype=np.complex128)
    iu = 1j * uu
    d = np.sqrt((rho * sigma * iu - kappa) ** 2 + sigma**2 * (iu + uu**2))
    g = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d)
    edt = np.exp(-d * t)
    c_h = (kappa * theta / sigma**2) * (
        (kappa - rho * sigma * iu - d) * t - 2.0 * np.log((1.0 - g * edt) / (1.0 - g))
    ) + iu * (r * t + np.log(s0))
    d_h = ((kappa - rho * sigma * iu - d) / sigma**2) * ((1.0 - edt) / (1.0 - g * edt))
    m_u = np.exp(iu * muj - uu**2 * sigj**2 / 2.0) - 1.0
    kappa_j = np.exp(muj + sigj**2 / 2.0) - 1.0
    jump = np.exp(lam * t * (m_u - iu * kappa_j))
    phi = np.asarray(np.exp(c_h + d_h * v0) * jump, dtype=np.complex128)
    if np.isscalar(u):
        return complex(phi.item())
    return phi


def _heston_prob(
    k: float,
    s0: float,
    t: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    lam: float,
    muj: float,
    sigj: float,
    j: int,
) -> float:
    """Risk-neutral probability P_j via Gauss quadrature."""

    def integrand(u: float) -> float:
        iu = 1j * u
        phi = bates_cf(u, s0, t, r, v0, kappa, theta, sigma, rho, lam, muj, sigj)
        phi2 = (
            bates_cf(u - 1j, s0, t, r, v0, kappa, theta, sigma, rho, lam, muj, sigj)
            if j == 1
            else bates_cf(u, s0, t, r, v0, kappa, theta, sigma, rho, lam, muj, sigj)
        )
        num = np.exp(-iu * k) * (phi2 if j == 1 else phi)
        den = iu * (
            bates_cf(-1j, s0, t, r, v0, kappa, theta, sigma, rho, lam, muj, sigj) if j == 1 else 1.0
        )
        return float(np.real(num / den))

    val = float(quad(integrand, 0.0, 200.0, limit=200, epsabs=1e-10)[0])
    return 0.5 + val / np.pi


def bates_call(
    s0: float,
    k: float,
    t: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    lam: float,
    muj: float,
    sigj: float,
) -> float:
    """European call under Bates SVJ."""
    if k <= 0:
        raise ValueError("bad strike")
    x = np.log(k)
    p1 = _heston_prob(x, s0, t, r, v0, kappa, theta, sigma, rho, lam, muj, sigj, 1)
    p2 = _heston_prob(x, s0, t, r, v0, kappa, theta, sigma, rho, lam, muj, sigj, 2)
    return float(s0 * p1 - k * np.exp(-r * t) * p2)


def synth_bates(
    seed: int = 20261231 + 338,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC SVJ smile grid + Bates CF sanity pair."""
    rng = np.random.default_rng(seed)
    ks = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    lamv = rng.uniform(0.2, 0.6)
    return ks, np.array([lamv])


def bench_bates(seed: int = 20261231 + 338) -> dict[str, float]:
    ks, _ = synth_bates(seed=seed)
    p = dict(s0=1.0, t=0.25, r=0.0, v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.6)
    svj = dict(lam=0.5, muj=-0.1, sigj=0.15)
    price_svj = np.array([bates_call(k=k, **p, **svj) for k in ks])
    price_no_jump = np.array([bates_call(k=k, **p, lam=0.0, muj=0.0, sigj=0.0) for k in ks])
    # martingale check: |phi(-i)| = E[S_T]/S0 = 1
    phi_mi = complex(
        bates_cf(
            -1j,
            **{k: p[k] for k in ("s0", "t", "r", "v0", "kappa", "theta", "sigma", "rho")},
            **svj,
        )
    )
    mart_err = float(abs(phi_mi - 1.0))
    put_skew_uplift = float(price_svj[0] - price_no_jump[0])
    ok = mart_err < 1e-6 and put_skew_uplift > 0.001 and float(np.all(np.diff(price_svj) < 0))
    out: dict[str, float] = {
        "synthetic_bates_martingale_err": mart_err,
        "synthetic_bates_otm_put_uplift": put_skew_uplift,
        "synthetic_bates_atm_call": float(price_svj[2]),
        "synthetic_bates_monotone_strikes": float(np.all(np.diff(price_svj) < 0)),
        "synthetic_score": 1.0 if ok else 0.0,
    }
    return out
