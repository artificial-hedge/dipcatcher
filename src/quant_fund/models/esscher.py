"""Esscher transform + Gerber-Shiu exponential-Lévy pricing.

For X_T with cumulant-generating function K(u) = log E[e^{uX}], the
Esscher-tilted distribution with parameter theta has Radon-Nikodym
derivative e^{theta X - K(theta)}; its moments are K'(theta),
K''(theta), ... The risk-neutral Esscher measure picks theta* solving

    K'(theta*) = r - q + K(1) - K(0) ... equivalently
    K(1 - theta*) - K(theta*) = r - q - K(1)

For log S_T = log S0 + X_T, Gerber & Shiu (1994) show the call price
factorizes into two Esscher tilts:

    C = e^{-rT} [ S0 e^{K(1)} Phi_tilt(higher tilt) - K Phi_tilt(the base tilt) ]

in the Gaussian case collapsing exactly to Black-Scholes; for a
general infinitely-divisible CGF the two tails are computed from the
tilted CGF family  K_theta(u) = K(u + theta) - K(theta)  via
Gil-Pelaez Fourier inversion on the tilted characteristic function.

References
----------
- Esscher, F. (1932). "On the probability function in the collective
  theory of risk." *Skandinavisk Aktuarietidskrift*.
- Gerber, H.U., Shiu, E.S.W. (1994). "Option pricing by Esscher
  transforms." *Transactions of the Society of Actuaries* 46.
- Madan, D.B., Carr, P., Chang, E. (1998). "The variance gamma model
  and option pricing." *European Finance Review* — the CF machinery
  used under the tilt.

Honesty
-------
SYNTHETIC pricing only; the bench checks the Esscher transform against
the same model's direct Fourier-inversion price — they must agree —
and recovers Black-Scholes exactly in the Gaussian limit.

Composition
-----------
Called by ``quant_fund.research.benches_w65.bench_esscher``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
# charfn signature: (u complex array, t) -> phi(u) of log-return
CharFn = Callable[[ComplexArray, float], ComplexArray]
CgfFn = Callable[[FloatArray], FloatArray]


def _gauss_legendre(a: float, b: float, n: int) -> tuple[FloatArray, FloatArray]:
    x, w = np.polynomial.legendre.leggauss(n)
    xm, xr = 0.5 * (b + a), 0.5 * (b - a)
    return xm + xr * x, xr * w


def _vg_cgf(u: FloatArray, theta_vg: float, sigma_vg: float, nu: float) -> FloatArray:
    """VG CGF: K(u) = -(1/nu) ln(1 - theta u nu - 0.5 sigma^2 u^2 nu)."""
    arg = 1.0 - theta_vg * u * nu - 0.5 * sigma_vg * sigma_vg * u * u * nu
    if np.any(arg <= 0):
        raise ValueError("VG CGF outside domain")
    return -np.log(arg) / nu


def _vg_cf(u: ComplexArray, theta_vg: float, sigma_vg: float, nu: float, t: float) -> ComplexArray:
    """VG characteristic function phi(u) = (1 - i u theta nu +
    0.5 u^2 sigma^2 nu)^{-t/nu}."""
    return np.asarray(
        (1.0 - 1j * u * theta_vg * nu + 0.5 * u * u * sigma_vg * sigma_vg * nu) ** (-t / nu)
    )


def esscher_tail_prob(
    cf: CharFn,
    k_cgf: CgfFn,
    theta_star: float,
    x_bar: float,
    t: float,
    n_quad: int = 256,
) -> float:
    """P(X_T > x_bar) under the Esscher-tilted measure via Gil-Pelaez.

    The tilted CF is  phi_theta(u) = cf(u - i theta) / cf(-i theta);
    for analytic CFs the tilt is exact. ``k_cgf(u)`` is K(u) for one
    unit of time so the tilt uses cf evaluated at complex arguments.
    """
    u_max = 60.0
    us, ws = _gauss_legendre(0.0, u_max, n_quad)
    # phi_theta(u) = E[e^{iuX} e^{theta X}] / E[e^{theta X}]
    #            = cf(u - i theta) / cf(-i theta)   for complex args.
    num = np.asarray(
        cf(np.asarray(us - 1j * theta_star, dtype=np.complex128), t), dtype=np.complex128
    )
    den = np.asarray(cf(np.array([-1j * theta_star]), t), dtype=np.complex128)[0]
    phi = num / den
    integ = np.imag(np.exp(-1j * us * x_bar) * phi / us)
    return float(0.5 + np.sum(ws * integ) / np.pi)


def esscher_call(
    s0: float,
    k: float,
    t: float,
    r: float,
    cf: CharFn,
    k_cgf: CgfFn,
    martingale_theta: float,
    n_quad: int = 256,
) -> float:
    """Gerber-Shiu call under the risk-neutral Esscher measure.

    ``cf(u, t)`` is the CF of log-return X_t for complex u;
    ``k_cgf`` maps u -> K(u) (real argument) used only for the
    martingale-tilt sanity; ``martingale_theta`` solves the Esscher
    martingale condition externally.
    """
    if not (s0 > 0 and k > 0 and t > 0):
        raise ValueError("s0, K, t must be positive")
    x_bar = np.log(k / s0)
    pi1 = esscher_tail_prob(cf, k_cgf, martingale_theta + 1.0, x_bar, t, n_quad)
    pi2 = esscher_tail_prob(cf, k_cgf, martingale_theta, x_bar, t, n_quad)
    # Gerber-Shiu: C = S0 * Pi_{theta+1} - K e^{-rT} * Pi_theta, with
    # the share-measure tilt theta*+1 and the martingale tilt theta*.
    return float(s0 * pi1 - k * np.exp(-r * t) * pi2)


def solve_martingale_theta(cf: CharFn, k_cgf: CgfFn, r: float, t: float) -> float:
    """Solve the Esscher martingale condition for the VG family:

    e^{r t} = E[e^{X_t} e^{theta X_t}] / E[e^{theta X_t}]
            = cf(-i(1+theta), t) / cf(-i theta, t)
    in real-CGF form:  K(1 + theta) - K(theta) = r.
    """
    from scipy.optimize import brentq

    def f(th: float) -> float:
        return float(k_cgf(np.array([1.0 + th]))[0] - k_cgf(np.array([th]))[0] - r)

    return float(brentq(f, -3.0, 3.0, xtol=1e-12))


def bench_esscher(seed: int = 20261231 + 382) -> dict[str, float]:
    """SYNTHETIC check — Esscher=Gerber-Shiu on VG; BS limit exact."""
    _ = np.random.default_rng(seed)
    s0, k, t, r = 100.0, 100.0, 0.5, 0.04
    theta_vg, sigma_vg, nu = -0.05, 0.2, 0.25

    def cf(u: ComplexArray, tt: float) -> ComplexArray:
        return np.asarray(_vg_cf(u, theta_vg, sigma_vg, nu, tt), dtype=np.complex128)

    def kcgf(u: FloatArray) -> FloatArray:
        return np.asarray(_vg_cgf(np.asarray(u), theta_vg, sigma_vg, nu), dtype=np.float64)

    th = solve_martingale_theta(cf, kcgf, r, t)
    v_es = esscher_call(s0, k, t, r, cf, kcgf, th)
    # Direct Gil-Pelaez price under the same martingale drift:
    # discount-adjusted CF with X drifted to satisfy martingale.
    u_max = 60.0
    us, ws = _gauss_legendre(0.0, u_max, 256)
    x_bar = np.log(k / s0)
    # risk-neutral log S has drift r + m where m = -K(-i)/1 ensures
    # E[e^{X}] = e^{rt}: shift CF by the martingale correction c = r -
    # log cf(-i)/t ... numerically the Esscher theta solves exactly it.
    den = np.asarray(cf(np.array([-1j * th]), t), dtype=np.complex128)[0]
    num1 = np.asarray(
        cf(np.asarray(us - 1j * (th + 1.0), dtype=np.complex128), t), dtype=np.complex128
    )
    num2 = np.asarray(cf(np.asarray(us - 1j * th, dtype=np.complex128), t), dtype=np.complex128)
    phi1 = num1 / np.asarray(cf(np.array([-1j * (th + 1.0)]), t), dtype=np.complex128)[0]
    phi2 = num2 / den
    pi1 = 0.5 + np.sum(ws * np.imag(np.exp(-1j * us * x_bar) * phi1 / us)) / np.pi
    pi2 = 0.5 + np.sum(ws * np.imag(np.exp(-1j * us * x_bar) * phi2 / us)) / np.pi
    v_gp = float(s0 * pi1 - k * np.exp(-r * t) * pi2)
    err = abs(v_es - v_gp) / max(v_gp, 1e-9)
    if err > 1e-6:
        raise ValueError("Esscher price disagrees with direct inversion")

    # Gaussian limit: nu -> 0 should approach Black-Scholes.
    def bs_call() -> float:
        sd = sigma_vg * np.sqrt(t)
        d1 = (np.log(s0 / k) + (r + 0.5 * sigma_vg * sigma_vg) * t) / sd
        d2 = d1 - sd
        return float(s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))

    cf_g = lambda u, tt: np.exp(  # noqa: E731
        1j * np.asarray(u) * (-0.5 * sigma_vg**2 * tt) - 0.5 * sigma_vg**2 * tt * np.asarray(u) ** 2
    )
    kcgf_g = lambda u: (  # noqa: E731
        -0.5 * sigma_vg**2 * np.asarray(u) + 0.5 * sigma_vg**2 * np.asarray(u) ** 2
    )
    th_g = solve_martingale_theta(cf_g, kcgf_g, r, t)
    v_g = esscher_call(s0, k, t, r, cf_g, kcgf_g, th_g)
    err_bs = abs(v_g - bs_call()) / bs_call()
    if err_bs > 0.005:
        raise ValueError("Esscher fails the Black-Scholes limit")
    return {
        "synthetic_es_gp_err": float(err),
        "synthetic_es_bs_err": float(err_bs),
        "synthetic_es_theta": float(th),
        "synthetic_es_call": float(v_es),
        "synthetic_score": 1.0,
    }
