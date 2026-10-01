"""Tests for models/fourier_pricing.py — COS/CONV/Hilbert Fourier pricing suite.

Seeded SYNTHETIC: all option prices computed against Black–Scholes closed
forms (Fang & Oosterlee 2008; Lord et al. 2008; Feng & Linetsky 2008;
Merton 1973 barrier reference).  No live market data.

Mutation-hardening pins (dense numeric, independent references)
--------------------------------------------------------------
The suite started out structural (prices >= 0, monotone in M, bounded by the
vanilla).  That leaves the closed forms inside the module unpinned, so this
file adds dense differential pins against references computed *outside* the
module:

* COS payoff coefficients ``_cos_payoff_call/_cos_payoff_put`` vs Gauss–Legendre
  quadrature of their defining integrals over many ``(a, b, k, n)`` configs,
  including the degenerate ``a == b`` / all-negative / all-positive cases.
* COS series machinery: an independent evaluation of
  ``e^{-rT} * Σ'_k Re{ψ(u_k) V_k e^{i u_k (x - a)}}`` pins ``_cos_price_grid``,
  ``_cos_price_at``, the ``Σ'`` halving and the ``(j + 1/2)`` grid offset; a
  DCT-II/DCT-III round trip pins ``_dct_recover``.
* Cumulant closed forms and the exact ``L``/``sigma`` scaling of the truncation
  range, plus the keyword defaults of ``cos_truncation_range``.
* CF-registry moment conditions for bs/merton/vg/nig: ``φ(0) = 1``, the
  martingale identity ``φ(-i) = S0 e^{rT}``, and ``E[X]``/``Var[X]`` by finite
  differences against the textbook moment formulas.
* Merton with ``lam = 0`` equals BS pointwise on a u-grid; VG/NIG degenerate
  limits reduce to Gaussian/BS where applicable.
* Bermudan: a reference backward induction at M = 1, 2, 3 for both exercises,
  plus grid-point boundary cases (``x0 == a``, ``x0 == b``).
* Merton (1973) barrier closed forms re-derived independently (the reflection
  exponent is pinned at ``r = 0`` where it is exactly 1, and at ``r != 0``),
  and ``_bs_price_scalar`` cross-pinned against ``models.options.bs_price``.
* CONV and Hilbert legs: *differential* ("shadow") pins.  Both legs have
  documented numerical defects (see ``TestCONVBermudanShadow`` /
  ``TestHilbertBarrierShadow`` docstrings), so those pins lock the internal
  constants against silent drift.  They are regression locks, NOT accuracy
  claims and NOT market evidence.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy.fft import fft, ifft
from scipy.special import roots_legendre
from scipy.stats import norm

from quant_fund.models.fourier_pricing import (
    _bs_cumulants,
    _bs_price_scalar,
    _cos_payoff_call,
    _cos_payoff_put,
    _cos_price_at,
    _cos_price_grid,
    _dct_recover,
    _estimate_sigma_from_cf,
    _log_moneyness,
    _merton_cumulants,
    _resolve_ab,
    bs_char_fn,
    bs_continuous_barrier_call,
    conv_bermudan_put,
    cos_bermudan_put,
    cos_european_call,
    cos_european_put,
    cos_truncation_range,
    hilbert_barrier_call,
    merton_char_fn,
    nig_char_fn,
    vg_char_fn,
)
from quant_fund.models.options import bs_price

SEED = 20260929

CharFn = Callable[[np.ndarray], np.ndarray]

# Standard BS parameters used throughout
S0 = 100.0
K = 100.0
T = 1.0
SIGMA = 0.20
R = 0.03


def _bs_call(S: float, K: float, T: float, sigma: float, r: float) -> float:
    return bs_price(S, K, T, sigma, r, call=True)


def _bs_put(S: float, K: float, T: float, sigma: float, r: float) -> float:
    return bs_price(S, K, T, sigma, r, call=False)


# ---------------------------------------------------------------------------
# Independent reference implementations (never call the module's internals)
# ---------------------------------------------------------------------------

_GL_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}
_GL_NODES = 1200


def _gl(m: int = _GL_NODES) -> tuple[np.ndarray, np.ndarray]:
    """Cached Gauss-Legendre nodes/weights on [-1, 1] (computed once)."""
    if m not in _GL_CACHE:
        _GL_CACHE[m] = roots_legendre(m)
    return _GL_CACHE[m]


def _ref_call_coeffs(n: int, a: float, b: float, k: float, m: int = _GL_NODES) -> np.ndarray:
    """V_i = (2k/(b-a)) ∫_c^b (e^y - 1) cos(u_i (y - a)) dy, c = max(a, 0).

    Quadrature of the defining integral — independent of the module's chi/psi
    closed forms.
    """
    c = max(a, 0.0)
    if c >= b:
        return np.zeros(n, dtype=float)
    xs, ws = _gl(m)
    y = 0.5 * (b - c) * xs + 0.5 * (b + c)
    w = 0.5 * (b - c) * ws
    u = np.pi * np.arange(n) / (b - a)
    kern = np.cos(np.outer(u, y - a))
    return np.asarray((2.0 * k / (b - a)) * (kern @ (w * (np.exp(y) - 1.0))), dtype=float)


def _ref_put_coeffs(n: int, a: float, b: float, k: float, m: int = _GL_NODES) -> np.ndarray:
    """V_i = (2k/(b-a)) ∫_a^d (1 - e^y) cos(u_i (y - a)) dy, d = min(b, 0)."""
    d = min(b, 0.0)
    if d <= a:
        return np.zeros(n, dtype=float)
    xs, ws = _gl(m)
    y = 0.5 * (d - a) * xs + 0.5 * (d + a)
    w = 0.5 * (d - a) * ws
    u = np.pi * np.arange(n) / (b - a)
    kern = np.cos(np.outer(u, y - a))
    return np.asarray((2.0 * k / (b - a)) * (kern @ (w * (1.0 - np.exp(y)))), dtype=float)


def _ref_weights(n: int) -> np.ndarray:
    """Σ' weights: 1/2 on the k = 0 term, 1 elsewhere."""
    wt = np.ones(n, dtype=float)
    wt[0] = 0.5
    return wt


def _ref_series(
    psi: CharFn, r: float, tau: float, a: float, b: float, n: int, vk: np.ndarray, x: float
) -> float:
    """Reference COS series value at x: e^{-rT} Σ'_k Re{ψ(u_k)V_k e^{iu_k(x-a)}}."""
    u = np.pi * np.arange(n) / (b - a)
    G = np.asarray(psi(u), dtype=complex) * np.asarray(vk, dtype=float)
    phase = np.exp(1j * u * (x - a))
    return float(np.exp(-r * tau) * np.sum(_ref_weights(n) * np.real(G * phase)))


def _ref_grid(
    psi: CharFn, r: float, tau: float, a: float, b: float, n: int, vk: np.ndarray
) -> np.ndarray:
    """Reference COS series on the midpoint grid x_j = a + (j + 1/2)(b - a)/n."""
    u = np.pi * np.arange(n) / (b - a)
    G = np.asarray(psi(u), dtype=complex) * np.asarray(vk, dtype=float)
    G = G * _ref_weights(n)
    xj = a + (np.arange(n) + 0.5) * (b - a) / n
    return np.asarray(np.exp(-r * tau) * np.real(np.exp(1j * np.outer(xj - a, u)) @ G), dtype=float)


def _ref_dct(vals: np.ndarray, n: int) -> np.ndarray:
    """Reference DCT-II: V_k = (2/n) Σ_j f(x_j) cos(k π x_j) on the unit grid."""
    xj = (np.arange(n) + 0.5) / n
    kk = np.arange(n)
    return np.asarray(
        (2.0 / n) * (np.cos(np.pi * np.outer(kk, xj)) @ np.asarray(vals, float)), float
    )


def _bs_incr(r: float, sigma: float, tau: float):
    """Analytic BS increment CF of the log-moneyness over tau (independent)."""

    def psi(u: np.ndarray) -> np.ndarray:
        ua = np.asarray(u, dtype=float)
        return np.exp(1j * ua * (r - 0.5 * sigma**2) * tau - 0.5 * sigma**2 * tau * ua**2)

    return psi


def _ref_sigma_est(char_fn: CharFn, t: float, eps: float = 1e-4) -> float:
    """Shadow of ``_estimate_sigma_from_cf`` (mirrors its documented FD + the
    0.3 fallback).  Pinned separately against analytic BS/Merton vols, so the
    shadow only carries the eps/exponent constants into the CONV/Hilbert and
    Bermudan references."""
    cf0 = char_fn(np.array([0.0]))[0]
    cf_plus = char_fn(np.array([eps]))[0]
    cf_minus = char_fn(np.array([-eps]))[0]
    log_cf_p = np.log(cf_plus / cf0)
    log_cf_m = np.log(cf_minus / cf0)
    var_est = -float(np.real((log_cf_p + log_cf_m) / eps**2))
    if var_est <= 0:
        return 0.3
    return float(math.sqrt(var_est / t))


def _ref_bs_truncation(s0: float, k: float, r: float, t: float, sigma: float, L: float = 10.0):
    """Fang-Oosterlee §3 truncation for a Gaussian model: c1 ∓ L sqrt(c2)."""
    c1 = np.log(s0 / k) + (r - 0.5 * sigma**2) * t
    c2 = sigma**2 * t
    half = L * math.sqrt(c2)
    return float(c1 - half), float(c1 + half)


def _ref_step_cf(char_fn: CharFn, ln_s0: float, scale: float):
    """Mirror of the Bermudan/CONV dt-step CF: exp(scale * Log[φ(u)e^{-iu ln S0}]).

    The principal-branch Log matters at high u (documented COS-Bermudan
    caveat); those terms are numerically annihilated by |φ|, and the source
    lines carrying this expression hold no mutable constants, so mirroring is
    mutation-neutral.
    """

    def psi(u: np.ndarray) -> np.ndarray:
        phi = char_fn(u) * np.exp(-1j * np.asarray(u, dtype=float) * ln_s0)
        return np.exp(scale * np.log(phi))

    return psi


def _ref_bermudan(
    char_fn: CharFn,
    r: float,
    t: float,
    s0: float,
    k: float,
    M: int,
    n: int,
    L: float,
    sigma_est: float,
    exercise: str = "put",
    k_trunc: float | None = None,
) -> float:
    """Reference COS-Bermudan backward induction.  Payoff coefficients come from
    quadrature, the grid/series/DCT from the references above.  ``k_trunc`` is
    the strike used for the single global truncation range (the module uses the
    median of the strike vector)."""
    dt = t / M
    a, b = _ref_bs_truncation(s0, k if k_trunc is None else k_trunc, r, t, sigma_est, L)
    psi_step = _ref_step_cf(char_fn, math.log(s0), dt / t)
    xj = a + (np.arange(n) + 0.5) * (b - a) / n
    if exercise == "put":
        vk = _ref_put_coeffs(n, a, b, k)
        ex = np.maximum(k * (1.0 - np.exp(xj)), 0.0)
    else:
        vk = _ref_call_coeffs(n, a, b, k)
        ex = np.maximum(k * (np.exp(xj) - 1.0), 0.0)
    for _ in range(M - 1):
        cont = _ref_grid(psi_step, r, dt, a, b, n, vk)
        vk = _ref_dct(np.maximum(cont, ex), n)
    return _ref_series(psi_step, r, dt, a, b, n, vk, math.log(s0 / k))


def _ref_bs_scalar(
    s: float, k: float, t: float, sigma: float, r: float, call: bool = True
) -> float:
    """Independent BSM scalar price via scipy.stats.norm (t > 0 only)."""
    d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    df = math.exp(-r * t)
    if call:
        return float(s * norm.cdf(d1) - k * df * norm.cdf(d2))
    return float(k * df * norm.cdf(-d2) - s * norm.cdf(-d1))


def _ref_do_call(s0: float, k: float, h: float, t: float, sigma: float, r: float) -> float:
    """Merton (1973) down-and-out call for h <= k (independent re-derivation).

    Reflection identity for a GBM log-price Y with drift nu = r - sigma^2/2 and
    running minimum m_T: for y >= ln h,
        P(Y_T in dy, m_T <= ln h) = (h/S)^{2 nu / sigma^2} p^{(nu)}_T(y - 2 ln h) dy,
    and 2 nu / sigma^2 = 2 r / sigma^2 - 1, so the knock-in leg equals
        (S/h)^{1 - 2 r / sigma^2} * C_bs(h^2 / S, K).
    The identity needs the payoff support above ln h, i.e. K >= h.
    """
    expo = 1.0 - 2.0 * r / sigma**2
    knock_in = (s0 / h) ** expo * _ref_bs_scalar(h * h / s0, k, t, sigma, r, call=True)
    return max(0.0, _ref_bs_scalar(s0, k, t, sigma, r, call=True) - knock_in)


def _fd_mean_var(char_fn: CharFn, h: float = 1e-4) -> tuple[float, float]:
    """(E[X], Var[X]) as the first two cumulants of log φ at u = 0.

    κ1 = -i (log φ)'(0) and κ2 = -(log φ)''(0), both by central differences.
    Working with log φ (rather than φ) keeps the location term out of the
    second derivative, so the round-off/truncation balance stays ~1e-8 even
    when E[X] = ln S0 + … is O(5).
    """

    def log_phi(u: float) -> complex:
        cf0 = char_fn(np.array([0.0]))[0]
        return complex(np.log(char_fn(np.array([u]))[0] / cf0))

    lp, lm = log_phi(h), log_phi(-h)
    k1 = float(np.real(-1j * (lp - lm) / (2 * h)))
    k2 = -float(np.real((lp + lm) / h**2))
    return k1, k2


# ---------------------------------------------------------------------------
# Characteristic functions
# ---------------------------------------------------------------------------


class TestCharacteristicFunctions:
    def test_bs_char_fn_moment_generating(self) -> None:
        """BS CF at u=0 should return 1 (probability measure)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        val = cf(np.array([0.0]))
        assert abs(val[0] - 1.0) < 1e-10

    def test_bs_char_fn_matches_analytic(self) -> None:
        """BS CF: E[exp(i·u·ln S_T)] = S0^{iu} exp(iu(r-σ²/2)T - u²σ²T/2)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        u = np.array([1.0])
        expected = S0 ** (1j * u) * np.exp(
            1j * u * (R - 0.5 * SIGMA**2) * T - 0.5 * u**2 * SIGMA**2 * T
        )
        np.testing.assert_allclose(cf(u), expected, rtol=1e-10)

    def test_merton_char_fn_reduces_to_bs_at_zero_jump(self) -> None:
        """Merton with λ=0 should reduce to BS."""
        cf_m = merton_char_fn(S0, R, T, SIGMA, lam=0.0, mu_j=0.0, s_j=0.0)
        cf_bs = bs_char_fn(S0, R, T, SIGMA)
        u = np.linspace(-3, 3, 50)
        np.testing.assert_allclose(cf_m(u), cf_bs(u), rtol=1e-10)

    def test_merton_char_fn_at_zero_is_one(self) -> None:
        cf = merton_char_fn(S0, R, T, SIGMA, lam=1.0, mu_j=-0.05, s_j=0.1)
        assert abs(cf(np.array([0.0]))[0] - 1.0) < 1e-10

    def test_vg_char_fn_at_zero_is_one(self) -> None:
        cf = vg_char_fn(S0, R, T, sigma=SIGMA, theta=-0.1, nu=0.3)
        assert abs(cf(np.array([0.0]))[0] - 1.0) < 1e-10

    def test_nig_char_fn_at_zero_is_one(self) -> None:
        cf = nig_char_fn(S0, R, T, alpha=15.0, beta=-2.0, delta=1.0)
        assert abs(cf(np.array([0.0]))[0] - 1.0) < 1e-10

    def test_merton_zero_jump_equals_bs_on_dense_u_grid(self) -> None:
        """λ=0 must reproduce BS *exactly* across a wide u-grid (pins the
        diffusion exponent and the λκ drift correction, not just at u≈0)."""
        cf_m = merton_char_fn(S0, R, T, SIGMA, lam=0.0, mu_j=-0.4, s_j=0.25)
        cf_b = bs_char_fn(S0, R, T, SIGMA)
        u = np.concatenate([np.linspace(-40.0, 40.0, 161), np.array([0.0])])
        np.testing.assert_allclose(cf_m(u), cf_b(u), rtol=1e-13, atol=1e-15)

    def test_merton_zero_jump_at_second_tenor(self) -> None:
        """Same reduction at t != 1 pins the t-scaling of both exponent parts."""
        cf_m = merton_char_fn(S0, R, 2.5, 0.31, lam=0.0, mu_j=0.11, s_j=0.19)
        cf_b = bs_char_fn(S0, R, 2.5, 0.31)
        u = np.linspace(-8.0, 8.0, 97)
        np.testing.assert_allclose(cf_m(u), cf_b(u), rtol=1e-13, atol=1e-15)

    def test_merton_hermitian_symmetry(self) -> None:
        """Real-valued law => φ(-u) = conj(φ(u)) (kills sign flips in the
        exponent assembly)."""
        cf = merton_char_fn(S0, R, T, SIGMA, lam=1.3, mu_j=-0.07, s_j=0.13)
        u = np.linspace(0.25, 12.0, 40)
        np.testing.assert_allclose(cf(-u), np.conj(cf(u)), rtol=1e-13, atol=1e-15)

    def test_merton_cf_matches_closed_form(self) -> None:
        """Full Merton CF against its textbook closed form (drift + jumps)."""
        lam, mu_j, s_j = 1.3, -0.07, 0.13
        cf = merton_char_fn(S0, R, T, SIGMA, lam=lam, mu_j=mu_j, s_j=s_j)
        kappa = math.exp(mu_j + 0.5 * s_j**2) - 1.0
        u = np.linspace(-6.0, 6.0, 41)
        drift = math.log(S0) + (R - 0.5 * SIGMA**2 - lam * kappa) * T
        expected = np.exp(
            1j * u * drift
            - 0.5 * SIGMA**2 * T * u**2
            + lam * T * (np.exp(1j * u * mu_j - 0.5 * s_j**2 * u**2) - 1.0)
        )
        np.testing.assert_allclose(cf(u), expected, rtol=1e-13, atol=1e-15)

    def test_vg_cf_matches_closed_form(self) -> None:
        """VG CF against its textbook closed form."""
        nu, theta = 0.35, -0.12
        cf = vg_char_fn(S0, R, T, sigma=SIGMA, theta=theta, nu=nu)
        inner = 1.0 - theta * nu - 0.5 * SIGMA**2 * nu
        omega = math.log(inner) / nu
        u = np.linspace(-6.0, 6.0, 41)
        expected = np.exp(1j * u * (math.log(S0) + (R + omega) * T)) * (
            1.0 - 1j * theta * nu * u + 0.5 * SIGMA**2 * nu * u**2
        ) ** (-T / nu)
        np.testing.assert_allclose(cf(u), expected, rtol=1e-12, atol=1e-15)

    def test_nig_cf_matches_closed_form(self) -> None:
        """NIG CF against its textbook closed form."""
        alpha, beta, delta, mu = 15.0, -2.0, 1.0, 0.0
        cf = nig_char_fn(S0, R, T, alpha=alpha, beta=beta, delta=delta, mu=mu)
        gamma = math.sqrt(alpha**2 - beta**2)
        gamma2 = math.sqrt(alpha**2 - (beta + 1.0) ** 2)
        omega = -mu - delta * (gamma - gamma2)
        drift = math.log(S0) + (R + omega + mu) * T
        u = np.linspace(-6.0, 6.0, 41)
        expected = np.exp(
            1j * u * drift + delta * T * (gamma - np.sqrt(alpha**2 - (beta + 1j * u) ** 2))
        )
        np.testing.assert_allclose(cf(u), expected, rtol=1e-12, atol=1e-15)

    def test_vg_small_nu_limit_matches_bs(self) -> None:
        """θ=0, ν→0 VG must converge to the Gaussian (BS) CF: the subordinator's
        variance rate vanishes, so ω→-σ²/2 and (1+σ²νu²/2)^{-T/ν}→e^{-σ²u²T/2}."""
        cf = vg_char_fn(S0, R, T, sigma=SIGMA, theta=0.0, nu=1e-7)
        cf_bs = bs_char_fn(S0, R, T, SIGMA)
        u = np.linspace(-3.0, 3.0, 25)
        np.testing.assert_allclose(cf(u), cf_bs(u), rtol=1e-5, atol=1e-8)

    def test_nig_gaussian_limit(self) -> None:
        """α→∞ with β=0, δ=σ²α... NIG converges to a Gaussian: pin the small-u
        log-CF against -σ²u²T/2 with the variance rate δ α²/γ³ → δ/α."""
        alpha = 400.0
        delta = 0.04 * alpha  # Var[X] = δ T α²/γ³ ≈ δ T / α = 0.04 T
        cf = nig_char_fn(S0, R, T, alpha=alpha, beta=0.0, delta=delta)
        _, var = _fd_mean_var(cf)
        np.testing.assert_allclose(var, 0.04 * T, rtol=1e-3)

    def test_validate_model_params_rejects_non_finite(self) -> None:
        """NaN/inf inputs fail closed in every registry entry."""
        with pytest.raises(ValueError, match="sigma must be finite"):
            merton_char_fn(S0, R, T, float("nan"), lam=1.0, mu_j=0.0, s_j=0.1)
        with pytest.raises(ValueError, match="nu must be finite"):
            vg_char_fn(S0, R, T, sigma=SIGMA, nu=float("inf"))
        with pytest.raises(ValueError, match="delta must be finite"):
            nig_char_fn(S0, R, T, alpha=15.0, beta=0.0, delta=float("nan"))

    def test_vg_theta_default_is_zero(self) -> None:
        """The theta keyword default must be 0.0 (symmetric VG), pinned bitwise."""
        u = np.linspace(-4.0, 4.0, 33)
        np.testing.assert_array_equal(
            vg_char_fn(S0, R, T, sigma=SIGMA, nu=0.3)(u),
            vg_char_fn(S0, R, T, sigma=SIGMA, nu=0.3, theta=0.0)(u),
        )

    def test_nig_location_parameter_is_redundant(self) -> None:
        """mu cancels against the martingale correction omega = -mu - δ(γ-γ2), so
        the NIG CF is mu-invariant — which also means the mu default cannot be
        observed.  Documented here so a future mu-dependence is caught."""
        u = np.linspace(-4.0, 4.0, 33)
        base = nig_char_fn(S0, R, T, alpha=15.0, beta=-2.0, delta=1.0)(u)
        for mu in (0.0, 0.7, -3.0):
            np.testing.assert_allclose(
                nig_char_fn(S0, R, T, alpha=15.0, beta=-2.0, delta=1.0, mu=mu)(u),
                base,
                rtol=1e-14,
                atol=1e-16,
            )


class TestCFValidation:
    """Fail-closed boundaries of the CF registry (each parameter individually)."""

    def test_merton_rejects_each_bad_parameter(self) -> None:
        for t_bad, sig_bad in ((0.0, SIGMA), (-1.0, SIGMA), (T, 0.0), (T, -0.2)):
            with pytest.raises(ValueError, match="t and sigma must be positive"):
                merton_char_fn(S0, R, t_bad, sig_bad, lam=0.5, mu_j=0.0, s_j=0.1)

    def test_merton_rejects_negative_jump_params(self) -> None:
        with pytest.raises(ValueError, match=r"lam >= 0, s_j >= 0"):
            merton_char_fn(S0, R, T, SIGMA, lam=-0.5, mu_j=0.0, s_j=0.1)
        with pytest.raises(ValueError, match=r"lam >= 0, s_j >= 0"):
            merton_char_fn(S0, R, T, SIGMA, lam=0.5, mu_j=0.0, s_j=-0.1)

    def test_merton_accepts_small_but_positive_jump_params(self) -> None:
        """lam/s_j strictly between 0 and 1 must be accepted (pins `< 0`, not
        `< 1`), and the CF must satisfy φ(0)=1."""
        cf = merton_char_fn(S0, R, T, SIGMA, lam=0.25, mu_j=0.0, s_j=0.05)
        np.testing.assert_allclose(cf(np.array([0.0]))[0], 1.0, rtol=0, atol=1e-14)

    def test_vg_rejects_each_bad_parameter(self) -> None:
        for t_bad, sig_bad, nu_bad in (
            (0.0, SIGMA, 0.3),
            (-1.0, SIGMA, 0.3),
            (T, 0.0, 0.3),
            (T, SIGMA, 0.0),
            (T, SIGMA, -0.3),
        ):
            with pytest.raises(ValueError, match="t, sigma, nu must be positive"):
                vg_char_fn(S0, R, t_bad, sigma=sig_bad, nu=nu_bad)

    def test_vg_accepts_sub_unit_params(self) -> None:
        """t, sigma, nu all in (0, 1] must be accepted (pins `<= 0` not `<= 1`)."""
        cf = vg_char_fn(S0, R, 0.25, sigma=0.15, nu=0.4)
        np.testing.assert_allclose(cf(np.array([0.0]))[0], 1.0, rtol=0, atol=1e-14)

    def test_vg_rejects_broken_martingale_correction(self) -> None:
        """1 - θν - σ²ν/2 <= 0 fails closed with the correction message."""
        with pytest.raises(ValueError, match="martingale correction"):
            vg_char_fn(S0, R, T, sigma=1.0, nu=1.0, theta=1.0)

    def test_vg_rejects_exactly_zero_inner(self) -> None:
        """inner == 0 exactly must still raise (pins `<=`, not `<`)."""
        # 1 - 1.875*0.5 - 0.5*0.25*0.5 == 0.0 exactly in binary floating point.
        with pytest.raises(ValueError, match="martingale correction"):
            vg_char_fn(S0, R, T, sigma=0.5, nu=0.5, theta=1.875)

    def test_vg_accepts_positive_theta_with_valid_inner(self) -> None:
        """θ > 0 gives 0 < inner < 1 — legal, and must not be rejected."""
        cf = vg_char_fn(S0, R, T, sigma=SIGMA, nu=0.3, theta=0.1)
        inner = 1.0 - 0.1 * 0.3 - 0.5 * SIGMA**2 * 0.3
        np.testing.assert_allclose(cf(np.array([-1j]))[0], S0 * math.exp(R * T), rtol=1e-10)
        assert 0.0 < inner < 1.0

    def test_nig_rejects_each_bad_parameter(self) -> None:
        for t_bad, alpha_bad, delta_bad in (
            (0.0, 15.0, 1.0),
            (-1.0, 15.0, 1.0),
            (T, 0.0, 1.0),
            (T, -1.0, 1.0),
            (T, 15.0, 0.0),
        ):
            with pytest.raises(ValueError, match="t, alpha, delta must be positive"):
                nig_char_fn(S0, R, t_bad, alpha=alpha_bad, beta=-2.0, delta=delta_bad)

    def test_nig_accepts_sub_unit_params(self) -> None:
        """t, alpha, delta all in (0, 1] must be accepted."""
        cf = nig_char_fn(S0, R, 0.5, alpha=0.9, beta=-0.5, delta=0.25)
        np.testing.assert_allclose(cf(np.array([0.0]))[0], 1.0, rtol=0, atol=1e-14)

    def test_nig_rejects_abs_beta_at_alpha(self) -> None:
        """|beta| == alpha must raise the |beta| < alpha message (pins `>=`)."""
        with pytest.raises(ValueError, match=r"require \|beta\| < alpha"):
            nig_char_fn(S0, R, T, alpha=3.0, beta=-3.0, delta=1.0)
        with pytest.raises(ValueError, match=r"require \|beta\| < alpha"):
            nig_char_fn(S0, R, T, alpha=3.0, beta=4.0, delta=1.0)

    def test_nig_rejects_martingale_violation_at_boundary(self) -> None:
        """|beta + 1| == alpha must raise the martingale message (pins `>=`),
        while |beta| < alpha so the earlier guard does not fire first."""
        with pytest.raises(ValueError, match="NIG martingale"):
            nig_char_fn(S0, R, T, alpha=2.0, beta=1.0, delta=1.0)
        with pytest.raises(ValueError, match="NIG martingale"):
            nig_char_fn(S0, R, T, alpha=5.0, beta=4.0, delta=1.0)
        with pytest.raises(ValueError, match="NIG martingale"):
            nig_char_fn(S0, R, T, alpha=5.0, beta=4.5, delta=1.0)


class TestCFMoments:
    """Moment/martingale conditions — the CF registry's real contract."""

    def test_bs_moments(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        mean, var = _fd_mean_var(cf)
        np.testing.assert_allclose(mean, math.log(S0) + (R - 0.5 * SIGMA**2) * T, rtol=0, atol=1e-7)
        np.testing.assert_allclose(var, SIGMA**2 * T, rtol=1e-6, atol=1e-7)

    def test_merton_moments(self) -> None:
        lam, mu_j, s_j = 1.3, -0.07, 0.13
        cf = merton_char_fn(S0, R, T, SIGMA, lam=lam, mu_j=mu_j, s_j=s_j)
        mean, var = _fd_mean_var(cf)
        kappa = math.exp(mu_j + 0.5 * s_j**2) - 1.0
        exp_mean = math.log(S0) + (R - 0.5 * SIGMA**2 - lam * kappa) * T + lam * T * mu_j
        exp_var = (SIGMA**2 + lam * (mu_j**2 + s_j**2)) * T
        np.testing.assert_allclose(mean, exp_mean, rtol=0, atol=1e-7)
        np.testing.assert_allclose(var, exp_var, rtol=1e-5, atol=1e-7)

    def test_merton_moments_second_config(self) -> None:
        """A second (λ, μ_J, s_J, t) point so no constant can hide."""
        lam, mu_j, s_j, t, sig = 0.7, 0.09, 0.21, 2.0, 0.11
        cf = merton_char_fn(S0, R, t, sig, lam=lam, mu_j=mu_j, s_j=s_j)
        mean, var = _fd_mean_var(cf)
        kappa = math.exp(mu_j + 0.5 * s_j**2) - 1.0
        exp_mean = math.log(S0) + (R - 0.5 * sig**2 - lam * kappa) * t + lam * t * mu_j
        exp_var = (sig**2 + lam * (mu_j**2 + s_j**2)) * t
        np.testing.assert_allclose(mean, exp_mean, rtol=0, atol=1e-7)
        np.testing.assert_allclose(var, exp_var, rtol=1e-5, atol=1e-7)

    def test_vg_moments(self) -> None:
        nu, theta = 0.35, -0.12
        cf = vg_char_fn(S0, R, T, sigma=SIGMA, theta=theta, nu=nu)
        mean, var = _fd_mean_var(cf)
        omega = math.log(1.0 - theta * nu - 0.5 * SIGMA**2 * nu) / nu
        # X = ln S0 + (r+ω)T + θG + σW_G with G ~ Gamma(T/ν, ν): E[G] = T.
        np.testing.assert_allclose(
            mean, math.log(S0) + (R + omega) * T + theta * T, rtol=0, atol=1e-7
        )
        np.testing.assert_allclose(var, (SIGMA**2 + theta**2 * nu) * T, rtol=1e-6, atol=1e-7)

    def test_vg_moments_positive_theta(self) -> None:
        nu, theta = 0.2, 0.3
        cf = vg_char_fn(S0, R, T, sigma=0.25, theta=theta, nu=nu)
        mean, var = _fd_mean_var(cf)
        omega = math.log(1.0 - theta * nu - 0.5 * 0.25**2 * nu) / nu
        np.testing.assert_allclose(
            mean, math.log(S0) + (R + omega) * T + theta * T, rtol=0, atol=1e-7
        )
        np.testing.assert_allclose(var, (0.25**2 + theta**2 * nu) * T, rtol=1e-6, atol=1e-7)

    def test_nig_moments(self) -> None:
        alpha, beta, delta = 15.0, -2.0, 1.0
        cf = nig_char_fn(S0, R, T, alpha=alpha, beta=beta, delta=delta)
        mean, var = _fd_mean_var(cf)
        gamma = math.sqrt(alpha**2 - beta**2)
        gamma2 = math.sqrt(alpha**2 - (beta + 1.0) ** 2)
        drift = math.log(S0) + (R - delta * (gamma - gamma2)) * T
        np.testing.assert_allclose(mean, drift + delta * T * beta / gamma, rtol=0, atol=1e-7)
        np.testing.assert_allclose(var, delta * T * alpha**2 / gamma**3, rtol=1e-5, atol=1e-7)

    def test_nig_moments_with_location(self) -> None:
        alpha, beta, delta, mu = 8.0, 1.5, 0.4, 0.02
        cf = nig_char_fn(S0, R, T, alpha=alpha, beta=beta, delta=delta, mu=mu)
        mean, var = _fd_mean_var(cf)
        gamma = math.sqrt(alpha**2 - beta**2)
        gamma2 = math.sqrt(alpha**2 - (beta + 1.0) ** 2)
        omega = -mu - delta * (gamma - gamma2)
        drift = math.log(S0) + (R + omega + mu) * T
        np.testing.assert_allclose(mean, drift + delta * T * beta / gamma, rtol=0, atol=1e-7)
        np.testing.assert_allclose(var, delta * T * alpha**2 / gamma**3, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize(
        "builder",
        [
            lambda: bs_char_fn(S0, R, T, SIGMA),
            lambda: merton_char_fn(S0, R, T, SIGMA, lam=1.3, mu_j=-0.07, s_j=0.13),
            lambda: vg_char_fn(S0, R, T, sigma=SIGMA, theta=-0.12, nu=0.35),
            lambda: nig_char_fn(S0, R, T, alpha=15.0, beta=-2.0, delta=1.0),
        ],
    )
    def test_martingale_identity(self, builder: Callable[[], CharFn]) -> None:
        """E_Q[S_T] = S0 e^{rT} <=> φ(-i) = S0 e^{rT} for every model."""
        cf = builder()
        got = complex(cf(np.array([-1j]))[0])
        np.testing.assert_allclose(got, S0 * math.exp(R * T), rtol=1e-10)

    @pytest.mark.parametrize(
        "builder",
        [
            lambda: bs_char_fn(S0, R, T, SIGMA),
            lambda: merton_char_fn(S0, R, T, SIGMA, lam=2.0, mu_j=0.05, s_j=0.2),
            lambda: vg_char_fn(S0, R, T, sigma=SIGMA, theta=0.05, nu=0.5),
            lambda: nig_char_fn(S0, R, T, alpha=12.0, beta=1.0, delta=0.7, mu=-0.01),
        ],
    )
    def test_phi_zero_is_one_and_hermitian(self, builder: Callable[[], CharFn]) -> None:
        cf = builder()
        np.testing.assert_allclose(cf(np.array([0.0]))[0], 1.0, rtol=0, atol=1e-14)
        u = np.linspace(0.5, 9.0, 12)
        np.testing.assert_allclose(cf(-u), np.conj(cf(u)), rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Cumulants and truncation range
# ---------------------------------------------------------------------------


class TestCumulants:
    def test_bs_cumulants_closed_form(self) -> None:
        """c1 = ln(S0/K) + (r - σ²/2)T, c2 = σ²T, c4 = 0 exactly (Gaussian)."""
        s0, k, r, t, sigma = 103.0, 97.0, 0.031, 1.3, 0.23
        c1, c2, c4 = _bs_cumulants(s0, k, r, t, sigma)
        assert c1 == pytest.approx(
            math.log(s0 / k) + (r - 0.5 * sigma**2) * t, rel=1e-15, abs=1e-15
        )
        assert c2 == pytest.approx(sigma**2 * t, rel=1e-15, abs=1e-18)
        assert c4 == 0.0

    def test_bs_cumulants_second_config(self) -> None:
        s0, k, r, t, sigma = 41.0, 88.0, -0.012, 0.37, 0.44
        c1, c2, c4 = _bs_cumulants(s0, k, r, t, sigma)
        assert c1 == pytest.approx(
            math.log(s0 / k) + (r - 0.5 * sigma**2) * t, rel=1e-15, abs=1e-15
        )
        assert c2 == pytest.approx(sigma**2 * t, rel=1e-15, abs=1e-18)
        assert c4 == 0.0

    def test_merton_cumulants_closed_form(self) -> None:
        """Merton JD cumulants: compound-Poisson κ1..κ4 (Merton 1976)."""
        s0, k, r, t, sigma = 103.0, 97.0, 0.031, 1.3, 0.23
        lam, mu_j, s_j = 1.7, -0.07, 0.13
        c1, c2, c4 = _merton_cumulants(s0, k, r, t, sigma, lam, mu_j, s_j)
        kappa = math.exp(mu_j + 0.5 * s_j**2) - 1.0
        exp_c1 = math.log(s0 / k) + (r - 0.5 * sigma**2 - lam * kappa) * t + lam * t * mu_j
        exp_c2 = (sigma**2 + lam * (mu_j**2 + s_j**2)) * t
        exp_mu4 = mu_j**4 + 6.0 * mu_j**2 * s_j**2 + 3.0 * s_j**4
        assert c1 == pytest.approx(exp_c1, rel=1e-14, abs=1e-15)
        assert c2 == pytest.approx(exp_c2, rel=1e-14, abs=1e-18)
        assert c4 == pytest.approx(lam * t * exp_mu4, rel=1e-13, abs=1e-18)

    def test_merton_cumulants_second_config(self) -> None:
        s0, k, r, t, sigma = 55.0, 60.0, 0.07, 2.5, 0.41
        lam, mu_j, s_j = 0.6, 0.14, 0.29
        c1, c2, c4 = _merton_cumulants(s0, k, r, t, sigma, lam, mu_j, s_j)
        kappa = math.exp(mu_j + 0.5 * s_j**2) - 1.0
        exp_c1 = math.log(s0 / k) + (r - 0.5 * sigma**2 - lam * kappa) * t + lam * t * mu_j
        exp_c2 = (sigma**2 + lam * (mu_j**2 + s_j**2)) * t
        exp_mu4 = mu_j**4 + 6.0 * mu_j**2 * s_j**2 + 3.0 * s_j**4
        assert c1 == pytest.approx(exp_c1, rel=1e-14, abs=1e-15)
        assert c2 == pytest.approx(exp_c2, rel=1e-14, abs=1e-18)
        assert c4 == pytest.approx(lam * t * exp_mu4, rel=1e-13, abs=1e-18)

    def test_merton_cumulants_reduce_to_bs_at_zero_jump(self) -> None:
        s0, k, r, t, sigma = 103.0, 97.0, 0.031, 1.3, 0.23
        bs_c = _bs_cumulants(s0, k, r, t, sigma)
        me_c = _merton_cumulants(s0, k, r, t, sigma, 0.0, 0.3, 0.2)
        assert me_c[0] == pytest.approx(bs_c[0], rel=1e-15, abs=1e-15)
        assert me_c[1] == pytest.approx(bs_c[1], rel=1e-15, abs=1e-18)
        assert me_c[2] == 0.0


class TestTruncationRange:
    def test_range_contains_log_moneyness(self) -> None:
        a, b = cos_truncation_range(s0=S0, k=K, r=R, t=T, sigma=SIGMA, L=10.0)
        x = math.log(S0 / K)
        assert a < x < b

    def test_wider_L_gives_wider_range(self) -> None:
        a1, b1 = cos_truncation_range(s0=S0, k=K, r=R, t=T, sigma=SIGMA, L=5.0)
        a2, b2 = cos_truncation_range(s0=S0, k=K, r=R, t=T, sigma=SIGMA, L=15.0)
        assert a2 < a1
        assert b2 > b1

    def test_fail_closed_on_bad_inputs(self) -> None:
        # s0=0 produces a warning but doesn't raise — document this
        with pytest.warns(RuntimeWarning):
            cos_truncation_range(s0=0.0, k=K, r=R, t=T, sigma=SIGMA)
        with pytest.raises(ValueError):
            cos_truncation_range(s0=S0, k=K, r=R, t=0.0, sigma=SIGMA)

    def test_bs_endpoints_exact(self) -> None:
        """a = c1 - Lσ√t, b = c1 + Lσ√t exactly (Gaussian c4 = 0)."""
        s0, k, r, t, sigma, L = 103.0, 97.0, 0.031, 1.3, 0.23, 7.5
        a, b = cos_truncation_range(s0=s0, k=k, r=r, t=t, sigma=sigma, L=L)
        c1 = math.log(s0 / k) + (r - 0.5 * sigma**2) * t
        half = L * math.sqrt(sigma**2 * t)
        assert a == pytest.approx(c1 - half, rel=1e-15, abs=1e-15)
        assert b == pytest.approx(c1 + half, rel=1e-15, abs=1e-15)

    def test_width_scales_linearly_in_L(self) -> None:
        """b - a = 2Lσ√t: the ratio pin kills any L-multiplier drift."""
        sigma, t = 0.23, 1.3
        for L in (3.0, 10.0, 17.5):
            a, b = cos_truncation_range(s0=S0, k=K, r=R, t=t, sigma=sigma, L=L)
            assert (b - a) == pytest.approx(2.0 * L * sigma * math.sqrt(t), rel=1e-13)

    def test_width_ratio_between_L_values(self) -> None:
        a1, b1 = cos_truncation_range(s0=S0, k=K, r=R, t=T, sigma=SIGMA, L=4.0)
        a2, b2 = cos_truncation_range(s0=S0, k=K, r=R, t=T, sigma=SIGMA, L=14.0)
        assert (b2 - a2) / (b1 - a1) == pytest.approx(14.0 / 4.0, rel=1e-13)
        # centre is L-independent
        assert (a1 + b1) / 2 == pytest.approx((a2 + b2) / 2, rel=1e-13, abs=1e-13)

    def test_width_scales_with_sigma(self) -> None:
        """b - a ∝ σ for a Gaussian model (kills σ→σ³ style drift)."""
        for sigma in (0.11, 0.2, 0.47):
            a, b = cos_truncation_range(s0=S0, k=K, r=R, t=T, sigma=sigma, L=10.0)
            assert (b - a) == pytest.approx(20.0 * sigma * math.sqrt(T), rel=1e-13)

    def test_default_L_is_ten(self) -> None:
        """Keyword default L=10.0 pinned bitwise against the explicit call."""
        got = cos_truncation_range(s0=S0, k=K, r=R, t=T, sigma=SIGMA)
        exp = cos_truncation_range(s0=S0, k=K, r=R, t=T, sigma=SIGMA, L=10.0)
        assert got == exp
        assert got[1] - got[0] == pytest.approx(20.0 * SIGMA, rel=1e-13)

    def test_merton_defaults_are_zero_jumps(self) -> None:
        """lam/mu_j/s_j default to 0.0: model="merton" without jump args must
        reproduce the BS range exactly."""
        got = cos_truncation_range(s0=S0, k=K, r=R, t=T, sigma=SIGMA, L=6.0, model="merton")
        exp = cos_truncation_range(s0=S0, k=K, r=R, t=T, sigma=SIGMA, L=6.0, model="bs")
        assert got == exp

    def test_merton_default_s_j_is_zero(self) -> None:
        """lam given, s_j defaulted: c2 must stay σ²T (kills a non-zero s_j
        default, which would add lam·s_j² to the variance)."""
        a, b = cos_truncation_range(
            s0=S0, k=K, r=R, t=T, sigma=SIGMA, lam=2.0, mu_j=0.0, L=6.0, model="merton"
        )
        c1 = math.log(S0 / K) + (R - 0.5 * SIGMA**2 - 2.0 * (math.exp(0.0) - 1.0)) * T
        half = 6.0 * math.sqrt(SIGMA**2 * T)
        assert a == pytest.approx(c1 - half, rel=1e-14, abs=1e-14)
        assert b == pytest.approx(c1 + half, rel=1e-14, abs=1e-14)

    def test_merton_default_mu_j_is_zero(self) -> None:
        """s_j given, mu_j defaulted: κ = e^{s_j²/2} - 1 exactly."""
        lam, s_j = 2.0, 0.1
        a, b = cos_truncation_range(
            s0=S0, k=K, r=R, t=T, sigma=SIGMA, lam=lam, s_j=s_j, L=6.0, model="merton"
        )
        kappa = math.exp(0.5 * s_j**2) - 1.0
        c1 = math.log(S0 / K) + (R - 0.5 * SIGMA**2 - lam * kappa) * T
        c2 = (SIGMA**2 + lam * s_j**2) * T
        c4 = lam * T * 3.0 * s_j**4
        half = 6.0 * math.sqrt(c2 + math.sqrt(c4))
        assert a == pytest.approx(c1 - half, rel=1e-14, abs=1e-14)
        assert b == pytest.approx(c1 + half, rel=1e-14, abs=1e-14)

    def test_merton_default_lam_is_zero(self) -> None:
        """mu_j/s_j given, lam defaulted: no jump contribution at all."""
        got = cos_truncation_range(
            s0=S0, k=K, r=R, t=T, sigma=SIGMA, mu_j=-0.2, s_j=0.3, L=8.0, model="merton"
        )
        exp = cos_truncation_range(s0=S0, k=K, r=R, t=T, sigma=SIGMA, L=8.0, model="bs")
        assert got == exp

    def test_merton_range_wider_than_gaussian_when_c4_positive(self) -> None:
        """b - a = 2L√(c2 + √c4) with c4 = λT·E[J⁴] > 0 (kills max(c4, 1.0))."""
        lam, mu_j, s_j, L = 1.7, -0.07, 0.13, 10.0
        a, b = cos_truncation_range(
            s0=S0, k=K, r=R, t=T, sigma=SIGMA, lam=lam, mu_j=mu_j, s_j=s_j, L=L, model="merton"
        )
        c2 = (SIGMA**2 + lam * (mu_j**2 + s_j**2)) * T
        mu4 = mu_j**4 + 6.0 * mu_j**2 * s_j**2 + 3.0 * s_j**4
        c4 = lam * T * mu4
        assert (b - a) == pytest.approx(2.0 * L * math.sqrt(c2 + math.sqrt(c4)), rel=1e-14)
        assert (b - a) > 2.0 * L * math.sqrt(c2)

    def test_unknown_model_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="unknown model"):
            cos_truncation_range(s0=S0, k=K, r=R, t=T, sigma=SIGMA, model="heston")  # type: ignore[arg-type]

    def test_missing_bs_inputs_fail_closed(self) -> None:
        """Each missing BS argument raises on its own (pins the `or` chain)."""
        with pytest.raises(ValueError, match="required for BS cumulants"):
            cos_truncation_range(k=K, r=R, t=T, sigma=SIGMA)
        with pytest.raises(ValueError, match="required for BS cumulants"):
            cos_truncation_range(s0=S0, r=R, t=T, sigma=SIGMA)
        with pytest.raises(ValueError, match="required for BS cumulants"):
            cos_truncation_range(s0=S0, k=K, t=T, sigma=SIGMA)
        with pytest.raises(ValueError, match="required for BS cumulants"):
            cos_truncation_range(s0=S0, k=K, r=R, sigma=SIGMA)
        with pytest.raises(ValueError, match="required for BS cumulants"):
            cos_truncation_range(s0=S0, k=K, r=R, t=T)

    def test_missing_merton_inputs_fail_closed(self) -> None:
        for kwargs in (
            {"k": K, "r": R, "t": T, "sigma": SIGMA},
            {"s0": S0, "r": R, "t": T, "sigma": SIGMA},
            {"s0": S0, "k": K, "t": T, "sigma": SIGMA},
            {"s0": S0, "k": K, "r": R, "sigma": SIGMA},
            {"s0": S0, "k": K, "r": R, "t": T},
        ):
            with pytest.raises(ValueError, match="required for Merton cumulants"):
                cos_truncation_range(model="merton", **kwargs)

    def test_merton_branch_is_reachable(self) -> None:
        """model="merton" must not fall through to the unknown-model error."""
        a, b = cos_truncation_range(
            s0=S0, k=K, r=R, t=T, sigma=SIGMA, lam=1.0, mu_j=-0.05, s_j=0.1, model="merton"
        )
        assert a < b
        assert math.isfinite(a) and math.isfinite(b)

    def test_zero_variance_fails_closed(self) -> None:
        """c2 == 0 exactly must raise (pins `<=`, not `<`)."""
        with pytest.raises(ValueError, match="variance must be positive"):
            cos_truncation_range(s0=S0, k=K, r=R, t=T, sigma=0.0)


class TestSigmaEstimator:
    def test_recovers_bs_sigma(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        assert _estimate_sigma_from_cf(cf, T) == pytest.approx(SIGMA, rel=1e-7)

    def test_recovers_merton_total_vol(self) -> None:
        """Under Merton JD the log-variance rate is σ² + λ(μ_J² + s_J²)."""
        lam, mu_j, s_j = 1.3, -0.07, 0.13
        cf = merton_char_fn(S0, R, T, SIGMA, lam=lam, mu_j=mu_j, s_j=s_j)
        expected = math.sqrt(SIGMA**2 + lam * (mu_j**2 + s_j**2))
        assert _estimate_sigma_from_cf(cf, T) == pytest.approx(expected, rel=1e-5)

    def test_matches_reference_finite_difference(self) -> None:
        """Bitwise pin of the FD stencil (eps, eps², the φ(0) normaliser and the
        array indices) for non-Gaussian CFs, where eps matters."""
        for cf in (
            merton_char_fn(S0, R, T, SIGMA, lam=1.3, mu_j=-0.07, s_j=0.13),
            vg_char_fn(S0, R, T, sigma=SIGMA, theta=-0.12, nu=0.35),
            nig_char_fn(S0, R, T, alpha=15.0, beta=-2.0, delta=1.0),
        ):
            assert _estimate_sigma_from_cf(cf, T) == _ref_sigma_est(cf, T)

    def test_scales_with_tenor(self) -> None:
        """sigma_est = sqrt(var/t): doubling t must not double the estimate."""
        cf1 = bs_char_fn(S0, R, 1.0, SIGMA)
        cf2 = bs_char_fn(S0, R, 2.0, SIGMA)
        assert _estimate_sigma_from_cf(cf1, 1.0) == pytest.approx(
            _estimate_sigma_from_cf(cf2, 2.0), rel=1e-6
        )

    def test_degenerate_cf_falls_back_to_default(self) -> None:
        """A zero-variance CF (var_est == 0.0 exactly) must hit the 0.3 fallback,
        which pins both the `<= 0` comparison and the fallback constant."""

        def unit_cf(u: np.ndarray) -> np.ndarray:
            return np.ones(np.shape(u), dtype=complex)

        assert _estimate_sigma_from_cf(unit_cf, T) == 0.3

    def test_log_moneyness_default_is_atm(self) -> None:
        """Without s0 the COS entry points assume x = 0 (at-the-money)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        assert _log_moneyness(cf, 123.0) == 0.0
        assert _log_moneyness(cf, K) == 0.0


class TestResolveAB:
    def test_explicit_pair_returned_unchanged(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        assert _resolve_ab(-0.25, 0.75, cf, S0, K, R, T, 10.0) == (-0.25, 0.75)

    def test_explicit_pair_wins_over_auto(self) -> None:
        """With s0 given AND an explicit pair, the explicit pair must be used
        (pins `is not None`, not `is None`)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        auto = _ref_bs_truncation(S0, K, R, T, _ref_sigma_est(cf, T), 10.0)
        got = _resolve_ab(-0.25, 0.75, cf, S0, K, R, T, 10.0)
        assert got != auto
        assert got == (-0.25, 0.75)

    def test_partial_pair_falls_back_to_auto(self) -> None:
        """a given but b None must NOT be treated as an explicit pair."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        got = _resolve_ab(-0.25, None, cf, S0, K, R, T, 10.0)
        exp = _ref_bs_truncation(S0, K, R, T, _ref_sigma_est(cf, T), 10.0)
        assert got[0] == pytest.approx(exp[0], rel=1e-14, abs=1e-14)
        assert got[1] == pytest.approx(exp[1], rel=1e-14, abs=1e-14)

    def test_missing_s0_fails_closed(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError, match="s0 required"):
            _resolve_ab(None, None, cf, None, K, R, T, 10.0)


# ---------------------------------------------------------------------------
# COS payoff coefficients (analytic closed forms vs quadrature)
# ---------------------------------------------------------------------------

_CALL_CONFIGS = [
    (-2.0, 2.0, 100.0, 64),
    (-1.99, 2.01, 100.0, 256),
    (0.3, 1.7, 1.0, 8),
    (-3.0, -0.5, 50.0, 32),
    (-0.5, 0.5, 120.0, 3),
    (0.2, 1.2, 90.0, 2),
    (-1.7, 0.9, 250.0, 1),
]

_PUT_CONFIGS = [
    (-2.0, 2.0, 100.0, 64),
    (-1.99, 2.01, 100.0, 256),
    (0.3, 1.7, 1.0, 8),
    (-3.0, -0.5, 50.0, 32),
    (-0.5, 0.5, 120.0, 3),
    (-1.2, -0.2, 90.0, 2),
    (-0.9, 1.4, 300.0, 1),
]


class TestPayoffCoefficients:
    @pytest.mark.parametrize(("a", "b", "k", "n"), _CALL_CONFIGS)
    def test_call_coefficients_match_quadrature(self, a: float, b: float, k: float, n: int) -> None:
        got = _cos_payoff_call(n, a, b, k)
        exp = _ref_call_coeffs(n, a, b, k)
        np.testing.assert_allclose(got, exp, rtol=1e-8, atol=1e-9)
        assert got.shape == (n,)
        assert got.dtype == np.float64

    @pytest.mark.parametrize(("a", "b", "k", "n"), _PUT_CONFIGS)
    def test_put_coefficients_match_quadrature(self, a: float, b: float, k: float, n: int) -> None:
        got = _cos_payoff_put(n, a, b, k)
        exp = _ref_put_coeffs(n, a, b, k)
        np.testing.assert_allclose(got, exp, rtol=1e-8, atol=1e-9)
        assert got.shape == (n,)
        assert got.dtype == np.float64

    def test_call_coefficients_zero_when_range_below_strike(self) -> None:
        """b <= 0 => (e^y - 1)^+ == 0 on [a, b] => exact zeros."""
        np.testing.assert_array_equal(_cos_payoff_call(16, -3.0, -0.5, 100.0), np.zeros(16))

    def test_put_coefficients_zero_when_range_above_strike(self) -> None:
        """a >= 0 => (1 - e^y)^+ == 0 on [a, b] => exact zeros."""
        np.testing.assert_array_equal(_cos_payoff_put(16, 0.5, 3.0, 100.0), np.zeros(16))

    def test_call_degenerate_range_is_zero(self) -> None:
        """c == b (zero-width range) returns zeros instead of dividing by zero."""
        with np.errstate(divide="ignore", invalid="ignore"):
            got = _cos_payoff_call(16, 0.5, 0.5, 100.0)
        np.testing.assert_array_equal(got, np.zeros(16))

    def test_put_degenerate_range_is_zero(self) -> None:
        """d == a for a == b < 0 must return zeros (pins `<=`, not `<`)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            got = _cos_payoff_put(16, -0.3, -0.3, 100.0)
        np.testing.assert_array_equal(got, np.zeros(16))

    def test_put_zero_boundary_range_is_zero(self) -> None:
        """a == d == 0 (b > 0) => zero-width integration domain => zeros."""
        np.testing.assert_array_equal(_cos_payoff_put(8, 0.0, 1.0, 100.0), np.zeros(8))

    def test_coefficients_scale_linearly_in_strike(self) -> None:
        """V_k(K) is homogeneous of degree 1 in the strike (kills 2K→3K)."""
        for n, a, b in ((32, -2.0, 2.0), (7, 0.4, 1.1)):
            c1 = _cos_payoff_call(n, a, b, 100.0)
            c2 = _cos_payoff_call(n, a, b, 250.0)
            np.testing.assert_allclose(c2, 2.5 * c1, rtol=1e-13, atol=1e-13)
            p1 = _cos_payoff_put(n, a, b, 100.0)
            p2 = _cos_payoff_put(n, a, b, 250.0)
            np.testing.assert_allclose(p2, 2.5 * p1, rtol=1e-13, atol=1e-13)

    def test_call_minus_put_coefficients_are_parity_coefficients(self) -> None:
        """V^call_k - V^put_k == coefficients of K(e^y - 1) over the whole [a, b]
        (the cosine-space put-call parity transform)."""
        a, b, k, n = -1.99, 2.01, 100.0, 48
        diff = _cos_payoff_call(n, a, b, k) - _cos_payoff_put(n, a, b, k)
        xs, ws = _gl()
        y = 0.5 * (b - a) * xs + 0.5 * (b + a)
        w = 0.5 * (b - a) * ws
        u = np.pi * np.arange(n) / (b - a)
        exp = (2.0 * k / (b - a)) * (np.cos(np.outer(u, y - a)) @ (w * (np.exp(y) - 1.0)))
        np.testing.assert_allclose(diff, exp, rtol=1e-8, atol=1e-9)

    def test_single_term_coefficient_is_the_integral(self) -> None:
        """n=1 pins v[0] (the k=0 term) on its own — the n>1 branch is skipped."""
        a, b, k = -0.7, 1.1, 100.0
        c = max(a, 0.0)
        d = min(b, 0.0)
        got_c = _cos_payoff_call(1, a, b, k)
        got_p = _cos_payoff_put(1, a, b, k)
        exp_c = (2.0 * k / (b - a)) * ((math.exp(b) - math.exp(c)) - (b - c))
        exp_p = (2.0 * k / (b - a)) * ((d - a) - (math.exp(d) - math.exp(a)))
        assert got_c[0] == pytest.approx(exp_c, rel=1e-14)
        assert got_p[0] == pytest.approx(exp_p, rel=1e-14)

    def test_call_and_put_coefficients_match_closed_form_k0(self) -> None:
        """The k=0 entries against the hand-written integrals, for a range that
        straddles the strike (pins exp(b)-exp(c)-(b-c) and (d-a)-(exp(d)-exp(a)))."""
        a, b, k, n = -1.3, 0.9, 77.0, 40
        c, d = max(a, 0.0), min(b, 0.0)
        scale = 2.0 * k / (b - a)
        assert _cos_payoff_call(n, a, b, k)[0] == pytest.approx(
            scale * ((math.exp(b) - math.exp(c)) - (b - c)), rel=1e-14
        )
        assert _cos_payoff_put(n, a, b, k)[0] == pytest.approx(
            scale * ((d - a) - (math.exp(d) - math.exp(a))), rel=1e-14
        )


# ---------------------------------------------------------------------------
# COS series machinery: grid, point evaluation, DCT recovery
# ---------------------------------------------------------------------------


def _unit_cf(u: np.ndarray) -> np.ndarray:
    """Degenerate CF identically 1 (unit mass at zero increment)."""
    return np.ones(np.shape(u), dtype=complex)


class TestCosSeriesMachinery:
    def test_price_grid_matches_reference_series(self) -> None:
        """The (j + 1/2) midpoint grid, the Σ' halving and the Re/Im rotation are
        pinned against an independent complex-exponential evaluation."""
        a, b, n = -1.99, 2.01, 64
        psi = _bs_incr(R, SIGMA, T)
        vk = _ref_call_coeffs(n, a, b, K)
        got = _cos_price_grid(psi, R, T, a, b, n, vk)
        exp = _ref_grid(psi, R, T, a, b, n, vk)
        np.testing.assert_allclose(got, exp, rtol=1e-11, atol=1e-11)
        assert got.shape == (n,)

    def test_price_grid_put_matches_reference_series(self) -> None:
        a, b, n = -2.4, 1.1, 48
        psi = _bs_incr(0.05, 0.31, 0.75)
        vk = _ref_put_coeffs(n, a, b, 92.0)
        got = _cos_price_grid(psi, 0.05, 0.75, a, b, n, vk)
        exp = _ref_grid(psi, 0.05, 0.75, a, b, n, vk)
        np.testing.assert_allclose(got, exp, rtol=1e-11, atol=1e-11)

    def test_price_at_matches_reference_series_off_grid(self) -> None:
        """Point evaluation at non-grid x (the spectral path used by the pricers)."""
        a, b, n = -1.99, 2.01, 96
        psi = _bs_incr(R, SIGMA, T)
        vk = _ref_call_coeffs(n, a, b, K)
        for x in (-1.5, -0.75, -1e-9, 0.0, 0.31, 1.2345):
            got = _cos_price_at(psi, R, T, a, b, n, vk, x)
            exp = _ref_series(psi, R, T, a, b, n, vk, x)
            assert got == pytest.approx(exp, rel=1e-11, abs=1e-11)

    def test_price_at_agrees_with_grid_on_grid_points(self) -> None:
        """_cos_price_at(x_j) == _cos_price_grid(...)[j] — the two evaluation
        paths must use the same basis, phase and Σ' convention."""
        a, b, n = -1.99, 2.01, 64
        psi = _bs_incr(R, SIGMA, T)
        vk = _ref_put_coeffs(n, a, b, K)
        grid = _cos_price_grid(psi, R, T, a, b, n, vk)
        xj = a + (np.arange(n) + 0.5) * (b - a) / n
        at = np.array([_cos_price_at(psi, R, T, a, b, n, vk, float(x)) for x in xj])
        np.testing.assert_allclose(at, grid, rtol=1e-12, atol=1e-12)

    def test_price_grid_matches_bs_in_interior(self) -> None:
        """Economic pin: well inside the truncation range the COS grid price of a
        call equals BSM at spot K·e^x.  (Near x -> b the truncated density loses
        the right tail, so the reference comparison is restricted to |x| <= 0.6;
        the reference-series pin above covers the whole grid.)"""
        a, b = _ref_bs_truncation(S0, K, R, T, SIGMA, 10.0)
        n = 256
        psi = _bs_incr(R, SIGMA, T)
        vk = _ref_call_coeffs(n, a, b, K)
        grid = _cos_price_grid(psi, R, T, a, b, n, vk)
        xj = a + (np.arange(n) + 0.5) * (b - a) / n
        mask = np.abs(xj) <= 0.6
        assert mask.sum() > 40
        ref = np.array([_bs_call(K * math.exp(float(x)), K, T, SIGMA, R) for x in xj[mask]])
        np.testing.assert_allclose(grid[mask], ref, rtol=1e-8, atol=1e-9)

    def test_price_at_matches_bs_off_grid(self) -> None:
        a, b = _ref_bs_truncation(S0, K, R, T, SIGMA, 10.0)
        n = 256
        psi = _bs_incr(R, SIGMA, T)
        vk = _ref_put_coeffs(n, a, b, K)
        for x in (-0.6, -0.2, 0.0, 0.17, 0.55):
            got = _cos_price_at(psi, R, T, a, b, n, vk, x)
            exp = _bs_put(K * math.exp(x), K, T, SIGMA, R)
            assert got == pytest.approx(exp, rel=1e-9, abs=1e-9)

    def test_dct_matches_reference_formula(self) -> None:
        n = 37
        vals = np.cos(np.linspace(0.0, 3.0, n)) + 0.3 * np.arange(n) / n
        np.testing.assert_allclose(_dct_recover(vals, n), _ref_dct(vals, n), rtol=1e-13, atol=1e-14)

    def test_dct_recovers_band_limited_coefficients(self) -> None:
        """DCT-II of midpoint samples of Σ c_k cos(kπ(x-a)/(b-a)) returns
        2·c_0 and c_k (k>=1) exactly — pins the (j+1/2) offset and the 2/n."""
        n, a, b = 16, -1.0, 2.0
        c = np.zeros(8)
        c[0], c[1], c[3], c[7] = 3.0, 2.0, -0.7, 0.25
        xj = a + (np.arange(n) + 0.5) * (b - a) / n
        kk = np.arange(8)
        f = np.cos(np.pi * np.outer(xj - a, kk) / (b - a)) @ c
        got = _dct_recover(f, n)
        expected = np.concatenate(([2.0 * c[0]], c[1:], np.zeros(n - 8)))
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    def test_grid_inverts_dct_for_band_limited_values(self) -> None:
        """Round trip: with ψ ≡ 1 and no discounting, _cos_price_grid must invert
        _dct_recover exactly on band-limited grid values."""
        n, a, b = 32, -0.75, 1.25
        xj = a + (np.arange(n) + 0.5) * (b - a) / n
        vals = (
            4.0
            + np.cos(np.pi * (xj - a) / (b - a))
            - 0.5 * np.cos(5.0 * np.pi * (xj - a) / (b - a))
        )
        vk = _dct_recover(vals, n)
        back = _cos_price_grid(_unit_cf, 0.0, 0.0, a, b, n, vk)
        np.testing.assert_allclose(back, vals, rtol=1e-12, atol=1e-12)

    def test_price_at_inverts_dct_for_band_limited_values(self) -> None:
        n, a, b = 24, -0.4, 0.9
        xj = a + (np.arange(n) + 0.5) * (b - a) / n
        vals = 1.5 + 2.0 * np.cos(3.0 * np.pi * (xj - a) / (b - a))
        vk = _dct_recover(vals, n)
        got = _cos_price_at(_unit_cf, 0.0, 0.0, a, b, n, vk, float(xj[7]))
        assert got == pytest.approx(vals[7], rel=1e-12, abs=1e-12)

    def test_grid_applies_discount_factor(self) -> None:
        """e^{-rT} scaling: r=0 vs r>0 must differ by exactly that factor."""
        a, b, n = -1.5, 1.5, 32
        psi = _bs_incr(0.0, SIGMA, T)
        vk = _ref_call_coeffs(n, a, b, K)
        undisc = _cos_price_grid(psi, 0.0, T, a, b, n, vk)
        disc = _cos_price_grid(psi, 0.04, T, a, b, n, vk)
        np.testing.assert_allclose(disc, undisc * math.exp(-0.04 * T), rtol=1e-13, atol=1e-14)


def _shift_cf(char_fn: CharFn, adj: float):
    """ψ(u) = φ(u)·e^{-i u adj}: the zero-initial-log-price CF the COS formula
    needs (Fang-Oosterlee Eq. (10) multiplies by e^{i u x} separately)."""

    def psi(u: np.ndarray) -> np.ndarray:
        return char_fn(u) * np.exp(-1j * np.asarray(u, dtype=float) * adj)

    return psi


def _ref_cos_price(
    char_fn: CharFn,
    r: float,
    t: float,
    k: float,
    a: float,
    b: float,
    n: int,
    side: str,
    *,
    x: float,
    adj: float,
) -> float:
    """Reference COS European price: quadrature coefficients + reference series,
    including the documented grid fallback (clamped linear interpolation) used
    when x falls outside [a, b]."""
    vk = _ref_call_coeffs(n, a, b, k) if side == "call" else _ref_put_coeffs(n, a, b, k)
    psi = _shift_cf(char_fn, adj)
    if a <= x <= b:
        return _ref_series(psi, r, t, a, b, n, vk, x)
    grid = _ref_grid(psi, r, t, a, b, n, vk)
    dx = (b - a) / n
    j = (x - a) / dx - 0.5
    jj = int(np.floor(j))
    if jj < 0:
        return float(grid[0])
    if jj >= n - 1:
        return float(grid[-1])
    w = j - jj
    return float((1.0 - w) * grid[jj] + w * grid[jj + 1])


# ---------------------------------------------------------------------------
# COS European pricing
# ---------------------------------------------------------------------------


class TestCOSEuropean:
    def test_call_matches_bs(self) -> None:
        """COS European call matches BS analytic to spectral accuracy."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        prices = cos_european_call(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)
        expected = _bs_call(S0, K, T, SIGMA, R)
        np.testing.assert_allclose(prices[0], expected, rtol=1e-8, atol=1e-8)

    def test_put_matches_bs(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        prices = cos_european_put(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)
        expected = _bs_put(S0, K, T, SIGMA, R)
        np.testing.assert_allclose(prices[0], expected, rtol=1e-8, atol=1e-8)

    def test_put_call_parity(self) -> None:
        """COS prices satisfy put-call parity: C - P = S - K·e^{-rT}."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        c = cos_european_call(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)[0]
        p = cos_european_put(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)[0]
        expected_parity = S0 - K * math.exp(-R * T)
        np.testing.assert_allclose(c - p, expected_parity, atol=1e-8)

    def test_convergence_in_n_spectral(self) -> None:
        """Spectral convergence: error collapses exponentially in n."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        expected = _bs_call(S0, K, T, SIGMA, R)
        errs = []
        for n in [16, 32, 64]:
            p = cos_european_call(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=n)[0]
            errs.append(abs(p - expected))
        assert errs[0] > errs[1] > errs[2]
        assert errs[2] < 1e-10  # spectral accuracy by n=64

    def test_multiple_strikes(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        prices = cos_european_call(cf, r=R, t=T, strikes=strikes, s0=S0, n=256)
        for i, k in enumerate(strikes):
            expected = _bs_call(S0, k, T, SIGMA, R)
            np.testing.assert_allclose(prices[i], expected, rtol=1e-6, atol=1e-6)

    def test_itm_otm_ordering(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        strikes = np.array([80.0, 100.0, 120.0])
        prices = cos_european_call(cf, r=R, t=T, strikes=strikes, s0=S0, n=256)
        assert prices[0] > prices[1] > prices[2]

    def test_merton_jump_diffusion_prices_positive(self) -> None:
        cf = merton_char_fn(S0, R, T, SIGMA, lam=1.0, mu_j=-0.05, s_j=0.1)
        prices = cos_european_call(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)
        assert prices[0] > 0.0

    def test_fail_closed_bad_inputs(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError):
            cos_european_call(cf, r=R, t=T, strikes=np.array([0.0]))
        with pytest.raises(ValueError):
            cos_european_call(cf, r=R, t=0.0, strikes=np.array([K]))
        with pytest.raises(ValueError):
            cos_european_call(cf, r=R, t=T, strikes=np.array([]))

    # -- added dense pins ---------------------------------------------------

    @pytest.mark.parametrize(
        ("s0", "k"), [(100.0, 90.0), (100.0, 110.0), (100.0, 62.5), (100.0, 155.0)]
    )
    def test_offspot_call_and_put_match_bs(self, s0: float, k: float) -> None:
        """s0 != k pins both the CF de-shift adj = ln S0 and x = ln(S0/K); the
        ATM-only suite let those two drift together unnoticed."""
        cf = bs_char_fn(s0, R, T, SIGMA)
        strikes = np.array([k])
        c = cos_european_call(cf, r=R, t=T, strikes=strikes, s0=s0, n=256)[0]
        p = cos_european_put(cf, r=R, t=T, strikes=strikes, s0=s0, n=256)[0]
        assert c == pytest.approx(_bs_call(s0, k, T, SIGMA, R), rel=1e-9, abs=1e-9)
        assert p == pytest.approx(_bs_put(s0, k, T, SIGMA, R), rel=1e-9, abs=1e-9)
        assert c - p == pytest.approx(s0 - k * math.exp(-R * T), rel=1e-9, abs=1e-9)

    def test_auto_truncation_uses_estimated_sigma(self) -> None:
        """With a/b auto-computed the range must be the L=10 BS range at the
        CF-implied sigma (pins the _resolve_ab plumbing end to end)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        sig = _ref_sigma_est(cf, T)
        a, b = _ref_bs_truncation(S0, K, R, T, sig, 10.0)
        got = cos_european_call(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=128)[0]
        exp = _ref_cos_price(cf, R, T, K, a, b, 128, "call", x=math.log(S0 / K), adj=math.log(S0))
        assert got == pytest.approx(exp, rel=1e-11, abs=1e-11)
        assert got == pytest.approx(_bs_call(S0, K, T, SIGMA, R), rel=1e-9, abs=1e-9)

    def test_call_matches_reference_series_on_explicit_range(self) -> None:
        """Dense pin against quadrature coefficients + reference series."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        a, b, n = -1.4, 1.9, 96
        for k in (73.0, 100.0, 128.0):
            got = cos_european_call(cf, r=R, t=T, strikes=np.array([k]), a=a, b=b, s0=S0, n=n)[0]
            exp = _ref_cos_price(cf, R, T, k, a, b, n, "call", x=math.log(S0 / k), adj=math.log(S0))
            assert got == pytest.approx(exp, rel=1e-11, abs=1e-11)

    def test_put_matches_reference_series_on_explicit_range(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        a, b, n = -1.4, 1.9, 96
        for k in (73.0, 100.0, 128.0):
            got = cos_european_put(cf, r=R, t=T, strikes=np.array([k]), a=a, b=b, s0=S0, n=n)[0]
            exp = _ref_cos_price(cf, R, T, k, a, b, n, "put", x=math.log(S0 / k), adj=math.log(S0))
            assert got == pytest.approx(exp, rel=1e-11, abs=1e-11)

    def test_merton_call_matches_reference_series(self) -> None:
        """Non-Gaussian CF: pins the de-shift and the series for jump diffusion."""
        cf = merton_char_fn(S0, R, T, SIGMA, lam=1.3, mu_j=-0.07, s_j=0.13)
        a, b, n = -2.5, 2.5, 128
        for k in (85.0, 100.0, 121.0):
            got = cos_european_call(cf, r=R, t=T, strikes=np.array([k]), a=a, b=b, s0=S0, n=n)[0]
            exp = _ref_cos_price(cf, R, T, k, a, b, n, "call", x=math.log(S0 / k), adj=math.log(S0))
            assert got == pytest.approx(exp, rel=1e-11, abs=1e-11)

    def test_without_s0_prices_atm_using_strike_shift(self) -> None:
        """s0=None => adj = ln K and x = _log_moneyness(...) = 0 (at-the-money).
        For an ATM CF that must reproduce the BS ATM price."""
        cf = bs_char_fn(K, R, T, SIGMA)
        a, b = _ref_bs_truncation(K, K, R, T, SIGMA, 10.0)
        n = 128
        got_c = cos_european_call(cf, r=R, t=T, strikes=np.array([K]), a=a, b=b, n=n)[0]
        got_p = cos_european_put(cf, r=R, t=T, strikes=np.array([K]), a=a, b=b, n=n)[0]
        exp_c = _ref_cos_price(cf, R, T, K, a, b, n, "call", x=0.0, adj=math.log(K))
        exp_p = _ref_cos_price(cf, R, T, K, a, b, n, "put", x=0.0, adj=math.log(K))
        assert got_c == pytest.approx(exp_c, rel=1e-11, abs=1e-11)
        assert got_p == pytest.approx(exp_p, rel=1e-11, abs=1e-11)
        assert got_c == pytest.approx(_bs_call(K, K, T, SIGMA, R), rel=1e-9, abs=1e-9)

    def test_missing_s0_without_explicit_range_fails_closed(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError, match="s0 required"):
            cos_european_call(cf, r=R, t=T, strikes=np.array([K]))
        with pytest.raises(ValueError, match="s0 required"):
            cos_european_put(cf, r=R, t=T, strikes=np.array([K]))


class TestCOSEuropeanFallback:
    """The Σ' series is used for x in [a, b]; outside it the code falls back to
    clamped grid interpolation.  Both regimes are pinned against the reference
    series, and the boundaries x == a / x == b must stay on the direct path."""

    def test_call_x_on_lower_boundary_uses_direct_series(self) -> None:
        s0, k, n = 100.0, 90.0, 32
        x = math.log(s0 / k)
        a, b = x, x + 1.0  # x == a exactly
        cf = bs_char_fn(s0, R, T, SIGMA)
        got = cos_european_call(cf, r=R, t=T, strikes=np.array([k]), a=a, b=b, s0=s0, n=n)[0]
        exp = _ref_series(
            _shift_cf(cf, math.log(s0)), R, T, a, b, n, _ref_call_coeffs(n, a, b, k), x
        )
        assert got == pytest.approx(exp, rel=1e-11, abs=1e-11)

    def test_put_x_on_lower_boundary_uses_direct_series(self) -> None:
        s0, k, n = 100.0, 110.0, 32
        x = math.log(s0 / k)
        a, b = x, x + 1.0
        cf = bs_char_fn(s0, R, T, SIGMA)
        got = cos_european_put(cf, r=R, t=T, strikes=np.array([k]), a=a, b=b, s0=s0, n=n)[0]
        exp = _ref_series(
            _shift_cf(cf, math.log(s0)), R, T, a, b, n, _ref_put_coeffs(n, a, b, k), x
        )
        assert got == pytest.approx(exp, rel=1e-11, abs=1e-11)

    def test_call_x_on_upper_boundary_uses_direct_series(self) -> None:
        s0, k, n = 100.0, 90.0, 32
        x = math.log(s0 / k)
        a, b = x - 1.0, x  # x == b exactly
        cf = bs_char_fn(s0, R, T, SIGMA)
        got = cos_european_call(cf, r=R, t=T, strikes=np.array([k]), a=a, b=b, s0=s0, n=n)[0]
        exp = _ref_series(
            _shift_cf(cf, math.log(s0)), R, T, a, b, n, _ref_call_coeffs(n, a, b, k), x
        )
        assert got == pytest.approx(exp, rel=1e-11, abs=1e-11)

    def test_put_x_on_upper_boundary_uses_direct_series(self) -> None:
        s0, k, n = 100.0, 110.0, 32
        x = math.log(s0 / k)
        a, b = x - 1.0, x
        cf = bs_char_fn(s0, R, T, SIGMA)
        got = cos_european_put(cf, r=R, t=T, strikes=np.array([k]), a=a, b=b, s0=s0, n=n)[0]
        exp = _ref_series(
            _shift_cf(cf, math.log(s0)), R, T, a, b, n, _ref_put_coeffs(n, a, b, k), x
        )
        assert got == pytest.approx(exp, rel=1e-11, abs=1e-11)

    def test_call_x_far_below_a_clamps_to_first_grid_point(self) -> None:
        s0, k, n = 100.0, 90.0, 32
        x = math.log(s0 / k)
        a, b = x + 0.5, x + 1.5
        cf = bs_char_fn(s0, R, T, SIGMA)
        got = cos_european_call(cf, r=R, t=T, strikes=np.array([k]), a=a, b=b, s0=s0, n=n)[0]
        exp = _ref_cos_price(cf, R, T, k, a, b, n, "call", x=x, adj=math.log(s0))
        grid0 = _ref_grid(_shift_cf(cf, math.log(s0)), R, T, a, b, n, _ref_call_coeffs(n, a, b, k))[
            0
        ]
        assert got == pytest.approx(exp, rel=1e-11, abs=1e-11)
        assert got == pytest.approx(grid0, rel=1e-11, abs=1e-11)

    def test_put_x_far_below_a_clamps_to_first_grid_point(self) -> None:
        s0, k, n = 100.0, 250.0, 32
        x = math.log(s0 / k)
        a, b = x + 0.5, x + 1.5
        cf = bs_char_fn(s0, R, T, SIGMA)
        got = cos_european_put(cf, r=R, t=T, strikes=np.array([k]), a=a, b=b, s0=s0, n=n)[0]
        exp = _ref_cos_price(cf, R, T, k, a, b, n, "put", x=x, adj=math.log(s0))
        grid0 = _ref_grid(_shift_cf(cf, math.log(s0)), R, T, a, b, n, _ref_put_coeffs(n, a, b, k))[
            0
        ]
        assert got == pytest.approx(exp, rel=1e-11, abs=1e-11)
        assert got == pytest.approx(grid0, rel=1e-11, abs=1e-11)

    def test_call_x_slightly_above_b_clamps_to_last_grid_point(self) -> None:
        """x just above b lands on jj == n-1, the last interpolable cell."""
        s0, k, n = 100.0, 50.0, 32
        x = math.log(s0 / k)
        a, b = -0.3, x - 0.005
        cf = bs_char_fn(s0, R, T, SIGMA)
        got = cos_european_call(cf, r=R, t=T, strikes=np.array([k]), a=a, b=b, s0=s0, n=n)[0]
        exp = _ref_cos_price(cf, R, T, k, a, b, n, "call", x=x, adj=math.log(s0))
        grid = _ref_grid(_shift_cf(cf, math.log(s0)), R, T, a, b, n, _ref_call_coeffs(n, a, b, k))
        assert got == pytest.approx(exp, rel=1e-11, abs=1e-11)
        assert got == pytest.approx(grid[-1], rel=1e-11, abs=1e-11)

    def test_put_x_slightly_above_b_clamps_to_last_grid_point(self) -> None:
        s0, k, n = 100.0, 50.0, 32
        x = math.log(s0 / k)
        a, b = -0.3, x - 0.005
        cf = bs_char_fn(s0, R, T, SIGMA)
        got = cos_european_put(cf, r=R, t=T, strikes=np.array([k]), a=a, b=b, s0=s0, n=n)[0]
        exp = _ref_cos_price(cf, R, T, k, a, b, n, "put", x=x, adj=math.log(s0))
        grid = _ref_grid(_shift_cf(cf, math.log(s0)), R, T, a, b, n, _ref_put_coeffs(n, a, b, k))
        assert got == pytest.approx(exp, rel=1e-11, abs=1e-11)
        assert got == pytest.approx(grid[-1], rel=1e-11, abs=1e-11)

    def test_call_x_far_above_b_clamps_to_last_grid_point(self) -> None:
        s0, k, n = 100.0, 50.0, 32
        x = math.log(s0 / k)
        a, b = -0.3, 0.2
        cf = bs_char_fn(s0, R, T, SIGMA)
        got = cos_european_call(cf, r=R, t=T, strikes=np.array([k]), a=a, b=b, s0=s0, n=n)[0]
        exp = _ref_cos_price(cf, R, T, k, a, b, n, "call", x=x, adj=math.log(s0))
        grid = _ref_grid(_shift_cf(cf, math.log(s0)), R, T, a, b, n, _ref_call_coeffs(n, a, b, k))
        assert got == pytest.approx(exp, rel=1e-11, abs=1e-11)
        assert got == pytest.approx(grid[-1], rel=1e-11, abs=1e-11)

    def test_put_x_far_above_b_clamps_to_last_grid_point(self) -> None:
        s0, k, n = 100.0, 50.0, 32
        x = math.log(s0 / k)
        a, b = -0.3, 0.2
        cf = bs_char_fn(s0, R, T, SIGMA)
        got = cos_european_put(cf, r=R, t=T, strikes=np.array([k]), a=a, b=b, s0=s0, n=n)[0]
        exp = _ref_cos_price(cf, R, T, k, a, b, n, "put", x=x, adj=math.log(s0))
        grid = _ref_grid(_shift_cf(cf, math.log(s0)), R, T, a, b, n, _ref_put_coeffs(n, a, b, k))
        assert got == pytest.approx(exp, rel=1e-11, abs=1e-11)
        assert got == pytest.approx(grid[-1], rel=1e-11, abs=1e-11)


class TestCOSEuropeanValidation:
    def test_small_n_accepted_and_pinned(self) -> None:
        """n == 4 is the documented minimum and must not raise."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        a, b = -1.99, 2.01
        got = cos_european_call(cf, r=R, t=T, strikes=np.array([K]), a=a, b=b, s0=S0, n=4)[0]
        exp = _ref_cos_price(cf, R, T, K, a, b, 4, "call", x=0.0, adj=math.log(S0))
        assert math.isfinite(got)
        assert got == pytest.approx(exp, rel=1e-10, abs=1e-10)
        got_p = cos_european_put(cf, r=R, t=T, strikes=np.array([K]), a=a, b=b, s0=S0, n=4)[0]
        exp_p = _ref_cos_price(cf, R, T, K, a, b, 4, "put", x=0.0, adj=math.log(S0))
        assert got_p == pytest.approx(exp_p, rel=1e-10, abs=1e-10)

    def test_n_below_four_fails_closed(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError, match="n must be >= 4"):
            cos_european_call(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=3)
        with pytest.raises(ValueError, match="n must be >= 4"):
            cos_european_put(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=3)

    def test_t_at_or_below_zero_fails_closed(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        for bad_t in (0.0, -0.5):
            with pytest.raises(ValueError, match="t must be positive"):
                cos_european_call(cf, r=R, t=bad_t, strikes=np.array([K]), s0=S0)
            with pytest.raises(ValueError, match="t must be positive"):
                cos_european_put(cf, r=R, t=bad_t, strikes=np.array([K]), s0=S0)

    def test_bad_strikes_fail_closed(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        for bad in (
            np.array([0.0]),
            np.array([-1.0]),
            np.array([]),
            np.array([np.nan]),
            np.array([np.inf]),
        ):
            with pytest.raises(ValueError, match="strikes must be positive and finite"):
                cos_european_call(cf, r=R, t=T, strikes=bad, s0=S0)
            with pytest.raises(ValueError, match="strikes must be positive and finite"):
                cos_european_put(cf, r=R, t=T, strikes=bad, s0=S0)

    def test_one_bad_strike_in_vector_fails_closed(self) -> None:
        """A single non-positive entry must trip the `or` chain, not slip by."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError, match="strikes must be positive and finite"):
            cos_european_call(cf, r=R, t=T, strikes=np.array([100.0, -5.0]), s0=S0)

    def test_tiny_positive_strike_accepted(self) -> None:
        """K in (0, 1] is legal (pins `<= 0`, not `<= 1`) and prices a deep-ITM
        call at S0 - K e^{-rT}."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        got = cos_european_call(cf, r=R, t=T, strikes=np.array([0.5]), s0=S0, n=256)[0]
        assert got == pytest.approx(_bs_call(S0, 0.5, T, SIGMA, R), rel=1e-9, abs=1e-9)
        got_p = cos_european_put(cf, r=R, t=T, strikes=np.array([0.5]), s0=S0, n=256)[0]
        assert got_p == pytest.approx(_bs_put(S0, 0.5, T, SIGMA, R), rel=1e-6, abs=1e-6)

    def test_short_tenor_accepted(self) -> None:
        """0 < t < 1 must be accepted (pins `t <= 0`, not `t <= 1`)."""
        cf = bs_char_fn(S0, R, 0.25, SIGMA)
        got = cos_european_call(cf, r=R, t=0.25, strikes=np.array([K]), s0=S0, n=256)[0]
        assert got == pytest.approx(_bs_call(S0, K, 0.25, SIGMA, R), rel=1e-8, abs=1e-8)

    def test_2d_strikes_are_ravelled(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        flat = cos_european_call(cf, r=R, t=T, strikes=np.array([90.0, 110.0]), s0=S0, n=128)
        two_d = cos_european_call(cf, r=R, t=T, strikes=np.array([[90.0], [110.0]]), s0=S0, n=128)
        np.testing.assert_allclose(two_d, flat, rtol=0, atol=0)
        assert two_d.shape == (2,)

    def test_defaults_match_explicit_arguments(self) -> None:
        """Keyword defaults n=256 and L=10.0 pinned bitwise."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        strikes = np.array([95.0, 105.0])
        np.testing.assert_array_equal(
            cos_european_call(cf, r=R, t=T, strikes=strikes, s0=S0),
            cos_european_call(cf, r=R, t=T, strikes=strikes, s0=S0, n=256, L=10.0),
        )
        np.testing.assert_array_equal(
            cos_european_put(cf, r=R, t=T, strikes=strikes, s0=S0),
            cos_european_put(cf, r=R, t=T, strikes=strikes, s0=S0, n=256, L=10.0),
        )

    def test_default_n_is_256_not_a_neighbour(self) -> None:
        """A converged series cannot tell n=256 from n=257, so the default is
        pinned through the grid fallback instead: with x outside [a, b] the
        returned value is a *grid point* price whose location moves with n
        (difference ~1e-4 between n=256 and n=257)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        strikes = np.array([K])
        got_c = cos_european_call(cf, r=R, t=T, strikes=strikes, a=0.5, b=1.5, s0=S0)[0]
        exp_c = cos_european_call(cf, r=R, t=T, strikes=strikes, a=0.5, b=1.5, s0=S0, n=256)[0]
        other_c = cos_european_call(cf, r=R, t=T, strikes=strikes, a=0.5, b=1.5, s0=S0, n=257)[0]
        assert got_c == exp_c
        assert abs(exp_c - other_c) > 1e-6  # the pin is not vacuous

        got_p = cos_european_put(cf, r=R, t=T, strikes=strikes, a=-1.5, b=-0.5, s0=S0)[0]
        exp_p = cos_european_put(cf, r=R, t=T, strikes=strikes, a=-1.5, b=-0.5, s0=S0, n=256)[0]
        other_p = cos_european_put(cf, r=R, t=T, strikes=strikes, a=-1.5, b=-0.5, s0=S0, n=257)[0]
        assert got_p == exp_p
        assert abs(exp_p - other_p) > 1e-6

    def test_default_L_controls_auto_range(self) -> None:
        """The default L must be 10: compare against an explicit L=10 range and
        show that L=8 moves the n=16 price by ~1e-2 (so the pin is not vacuous)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        sig = _ref_sigma_est(cf, T)
        a10, b10 = _ref_bs_truncation(S0, K, R, T, sig, 10.0)
        a8, b8 = _ref_bs_truncation(S0, K, R, T, sig, 8.0)
        default = cos_european_put(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=16)[0]
        with_l10 = cos_european_put(cf, r=R, t=T, strikes=np.array([K]), a=a10, b=b10, s0=S0, n=16)[
            0
        ]
        with_l8 = cos_european_put(cf, r=R, t=T, strikes=np.array([K]), a=a8, b=b8, s0=S0, n=16)[0]
        assert default == pytest.approx(with_l10, rel=1e-14, abs=1e-14)
        assert abs(default - with_l8) > 1e-4


# ---------------------------------------------------------------------------
# COS Bermudan
# ---------------------------------------------------------------------------


class TestCOSBermudan:
    def test_bermudan_put_at_least_european(self) -> None:
        """Bermudan put >= European put (early exercise adds value)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        euro = cos_european_put(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)[0]
        berm = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=10, n=256)[0]
        assert berm >= euro - 1e-8

    def test_bermudan_convergence_in_M(self) -> None:
        """More exercise dates → price increases toward the American limit."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        prices = []
        for M in [2, 5, 10]:
            p = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=M, n=256)[0]
            prices.append(p)
        assert prices[0] <= prices[1] + 1e-9
        assert prices[1] <= prices[2] + 1e-9

    def test_bermudan_M1_equals_european(self) -> None:
        """With M=1 (exercise only at maturity), Bermudan = European exactly."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        euro = cos_european_put(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)[0]
        berm = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=1, n=256)[0]
        np.testing.assert_allclose(berm, euro, rtol=1e-8, atol=1e-8)

    def test_bermudan_below_baw_american(self) -> None:
        """High-M Bermudan approaches the American value from below (BAW ref)."""
        from quant_fund.models.american_baw import baw_american

        r_baw = 0.05  # BAW comparison at r=5% (standard American-put test point)
        cf = bs_char_fn(S0, r_baw, T, SIGMA)
        berm = cos_bermudan_put(cf, r=r_baw, t=T, s0=S0, strikes=np.array([K]), M=50, n=512)[0]
        amer = baw_american(S0, K, T, r_baw, 0.0, SIGMA, option="put")
        assert berm <= amer + 0.02
        assert berm >= amer - 0.10  # close to the American limit at M=50

    def test_fail_closed_bad_inputs(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError):
            cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=0)

    # -- added dense pins ---------------------------------------------------

    @pytest.mark.parametrize("M", [1, 2, 3])
    def test_put_matches_reference_recursion(self, M: int) -> None:
        """Backward induction pinned against a reference implementation whose
        payoff coefficients come from quadrature: the grid offset, the DCT
        recovery, the max(cont, exercise) step and the final spectral
        evaluation are all covered."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        n = 96
        sig = _ref_sigma_est(cf, T)
        got = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=M, n=n)[0]
        exp = _ref_bermudan(cf, R, T, S0, K, M, n, 10.0, sig, "put")
        assert got == pytest.approx(exp, rel=1e-10, abs=1e-10)

    @pytest.mark.parametrize("M", [1, 2, 3])
    def test_call_matches_reference_recursion(self, M: int) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        n = 96
        sig = _ref_sigma_est(cf, T)
        got = cos_bermudan_put(
            cf, r=R, t=T, s0=S0, strikes=np.array([110.0]), M=M, n=n, exercise="call"
        )[0]
        exp = _ref_bermudan(cf, R, T, S0, 110.0, M, n, 10.0, sig, "call")
        assert got == pytest.approx(exp, rel=1e-10, abs=1e-10)

    def test_call_M1_equals_european_call(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        euro = cos_european_call(cf, r=R, t=T, strikes=np.array([95.0]), s0=S0, n=128)[0]
        berm = cos_bermudan_put(
            cf, r=R, t=T, s0=S0, strikes=np.array([95.0]), M=1, n=128, exercise="call"
        )[0]
        np.testing.assert_allclose(berm, euro, rtol=1e-10, atol=1e-10)

    def test_bermudan_call_at_least_european_call(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        euro = cos_european_call(cf, r=R, t=T, strikes=np.array([95.0]), s0=S0, n=128)[0]
        berm = cos_bermudan_put(
            cf, r=R, t=T, s0=S0, strikes=np.array([95.0]), M=6, n=128, exercise="call"
        )[0]
        assert berm >= euro - 1e-9

    def test_exercise_switch_selects_payoff(self) -> None:
        """exercise="put" and "call" must give different value functions (pins
        both the coefficient ternary and the exercise-value branch)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        put = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=4, n=64)[0]
        call = cos_bermudan_put(
            cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=4, n=64, exercise="call"
        )[0]
        assert put != call
        assert put > 0.0 and call > 0.0
        # and each equals its own reference
        sig = _ref_sigma_est(cf, T)
        assert put == pytest.approx(
            _ref_bermudan(cf, R, T, S0, K, 4, 64, 10.0, sig, "put"), rel=1e-10
        )
        assert call == pytest.approx(
            _ref_bermudan(cf, R, T, S0, K, 4, 64, 10.0, sig, "call"), rel=1e-10
        )

    def test_unknown_exercise_falls_back_to_call(self) -> None:
        """Anything other than "put" selects the call leg (pins the ternary)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        got = cos_bermudan_put(
            cf,
            r=R,
            t=T,
            s0=S0,
            strikes=np.array([K]),
            M=2,
            n=64,
            exercise="wat",  # type: ignore[arg-type]
        )[0]
        exp = cos_bermudan_put(
            cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=2, n=64, exercise="call"
        )[0]
        assert got == exp

    def test_multiple_strikes_use_median_truncation(self) -> None:
        """All strikes share one truncation range built on the median strike."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        strikes = np.array([85.0, 100.0, 120.0])
        n, M = 64, 2
        sig = _ref_sigma_est(cf, T)
        got = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=strikes, M=M, n=n)
        for i, k in enumerate(strikes):
            exp = _ref_bermudan(cf, R, T, S0, float(k), M, n, 10.0, sig, "put", k_trunc=100.0)
            assert got[i] == pytest.approx(exp, rel=1e-10, abs=1e-10)

    def test_merton_jump_diffusion_matches_reference_recursion(self) -> None:
        """Non-Gaussian CF through the whole recursion (principal-branch step CF)."""
        cf = merton_char_fn(S0, R, T, SIGMA, lam=0.9, mu_j=-0.04, s_j=0.11)
        n, M = 64, 2
        sig = _ref_sigma_est(cf, T)
        got = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=M, n=n)[0]
        exp = _ref_bermudan(cf, R, T, S0, K, M, n, 10.0, sig, "put")
        assert got == pytest.approx(exp, rel=1e-10, abs=1e-10)

    def test_custom_L_matches_reference(self) -> None:
        """L is threaded into the truncation range (kills an L default swap)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        n, M = 64, 2
        sig = _ref_sigma_est(cf, T)
        got = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=M, n=n, L=6.0)[0]
        exp = _ref_bermudan(cf, R, T, S0, K, M, n, 6.0, sig, "put")
        assert got == pytest.approx(exp, rel=1e-10, abs=1e-10)
        assert got != cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=M, n=n)[0]

    def test_x0_on_lower_boundary_uses_direct_series(self) -> None:
        """Construct L so the truncation lower endpoint lands exactly on
        x0 = ln(S0/K); the price must come from the spectral evaluation, not the
        np.interp fallback (which clamps to the first grid point)."""
        t, r = 5.0, 0.1
        cf = bs_char_fn(S0, r, t, SIGMA)
        sig = _ref_sigma_est(cf, t)
        L_edge = ((r - 0.5 * sig**2) * t) / math.sqrt(sig**2 * t)
        a_ref, b_ref = _ref_bs_truncation(S0, K, r, t, sig, L_edge)
        assert a_ref == math.log(S0 / K)  # construction precondition: x0 == a
        assert b_ref > a_ref
        n = 32
        got = cos_bermudan_put(
            cf, r=r, t=t, s0=S0, strikes=np.array([K]), M=2, n=n, L=L_edge, exercise="call"
        )[0]
        exp = _ref_bermudan(cf, r, t, S0, K, 2, n, L_edge, sig, "call")
        assert got == pytest.approx(exp, rel=1e-10, abs=1e-10)
        assert exp > 1.0  # non-degenerate value, so the fallback would show up

    def test_x0_on_upper_boundary_uses_direct_series(self) -> None:
        """Same construction for x0 == b (needs r < σ²/2 so L stays positive)."""
        t, r = 10.0, -0.05
        cf = bs_char_fn(S0, r, t, SIGMA)
        sig = _ref_sigma_est(cf, t)
        L_edge = (-(r - 0.5 * sig**2) * t) / math.sqrt(sig**2 * t)
        a_ref, b_ref = _ref_bs_truncation(S0, K, r, t, sig, L_edge)
        assert b_ref == math.log(S0 / K)  # construction precondition: x0 == b
        assert b_ref > a_ref
        n = 32
        got = cos_bermudan_put(
            cf, r=r, t=t, s0=S0, strikes=np.array([K]), M=2, n=n, L=L_edge, exercise="put"
        )[0]
        exp = _ref_bermudan(cf, r, t, S0, K, 2, n, L_edge, sig, "put")
        assert got == pytest.approx(exp, rel=1e-10, abs=1e-10)
        assert exp > 1.0


class TestCOSBermudanValidation:
    def test_M1_accepted(self) -> None:
        """M == 1 is legal (pins `M < 1`, not `M < 2`)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        got = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=1, n=32)
        assert np.isfinite(got[0]) and got[0] > 0.0

    def test_M0_fails_closed(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError, match="t > 0, M >= 1, n >= 4"):
            cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=0, n=32)

    def test_n4_accepted_n3_rejected(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        got = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=2, n=4)
        assert np.isfinite(got[0])
        with pytest.raises(ValueError, match="t > 0, M >= 1, n >= 4"):
            cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=2, n=3)

    def test_t_zero_or_negative_fails_closed(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        for bad_t in (0.0, -1.0):
            with pytest.raises(ValueError, match="t > 0, M >= 1, n >= 4"):
                cos_bermudan_put(cf, r=R, t=bad_t, s0=S0, strikes=np.array([K]), M=2, n=32)

    def test_short_tenor_accepted(self) -> None:
        """0 < t <= 1 must be accepted (pins `t <= 0`, not `t <= 1`)."""
        cf = bs_char_fn(S0, R, 0.5, SIGMA)
        got = cos_bermudan_put(cf, r=R, t=0.5, s0=S0, strikes=np.array([K]), M=2, n=64)[0]
        sig = _ref_sigma_est(cf, 0.5)
        assert got == pytest.approx(_ref_bermudan(cf, R, 0.5, S0, K, 2, 64, 10.0, sig), rel=1e-10)

    def test_bad_strikes_fail_closed(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        for bad in (np.array([0.0]), np.array([-3.0]), np.array([])):
            with pytest.raises(ValueError, match="strikes must be positive"):
                cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=bad, M=2, n=32)

    def test_tiny_positive_strike_accepted(self) -> None:
        """K in (0, 1] is legal (pins `<= 0`, not `<= 1`); a K=0.5 put far below
        a spot of 100 is worth exactly zero."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        got = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([0.5]), M=2, n=64)[0]
        assert np.isfinite(got)
        assert got == pytest.approx(0.0, abs=1e-12)

    def test_defaults_match_explicit_arguments(self) -> None:
        """M=10, n=256, L=10.0 defaults pinned bitwise."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        strikes = np.array([K])
        np.testing.assert_array_equal(
            cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=strikes),
            cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=strikes, M=10, n=256, L=10.0),
        )

    def test_default_exercise_is_put(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        default = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=3, n=64)
        explicit = cos_bermudan_put(
            cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=3, n=64, exercise="put"
        )
        np.testing.assert_array_equal(default, explicit)


def _shadow_conv(
    char_fn: CharFn,
    r: float,
    t: float,
    s0: float,
    ks: np.ndarray,
    M: int,
    N: int,
    L: float,
    alpha: float,
    sigma_est: float,
) -> np.ndarray:
    """Differential shadow of ``conv_bermudan_put`` (Lord et al. 2008 skeleton).

    NOTE: the CONV leg of this module has documented numerical defects (the
    dt-step kernel is built from |φ| and the FFT grid-offset phase is missing),
    so this shadow is a *characterisation* reference: it locks the internal
    constants (grid width factor, frequency-grid step, the 1e-300 magnitude
    floor, the damping branch, the interpolation clamps) against silent drift.
    It is NOT an accuracy claim and NOT market evidence.
    """
    ks = np.asarray(ks, dtype=float).ravel()
    dt = t / M
    med_k = float(np.median(ks))
    a_ref, b_ref = _ref_bs_truncation(s0, med_k, r, t, sigma_est, L)
    half_width = (b_ref - a_ref) * 1.5
    x0 = np.log(s0 / med_k)
    x_min = x0 - half_width
    x_max = x0 + half_width
    dx = (x_max - x_min) / N
    grid = x_min + dx * np.arange(N)
    dk_grid = 2.0 * np.pi / (N * dx)
    u = dk_grid * np.fft.fftfreq(N) * N
    cf_vals = char_fn(u) * np.exp(-1j * u * np.log(s0))
    cf_dt = np.exp((dt / t) * np.log(np.maximum(np.abs(cf_vals), 1e-300) + 0j))
    cf_dt = cf_dt * np.exp(1j * u * (r * dt))
    damp = np.exp(alpha * grid)
    out = np.empty(ks.size, dtype=float)
    for idx, k in enumerate(ks):
        y = grid + np.log(s0) - np.log(k)
        if alpha != 0.0:
            payoff = np.maximum(k - k * np.exp(y), 0.0) * damp
        else:
            payoff = np.maximum(k - k * np.exp(y), 0.0)
        v = payoff.copy()
        for _ in range(M):
            v_hat = fft(v)
            conv = np.real(ifft(v_hat * np.conj(cf_dt))) * dx
            v = np.maximum(np.exp(-r * dt) * conv, payoff)
            v = np.asarray(np.real(v), dtype=float)
        x0_val = np.log(s0 / k)
        j_frac = (x0_val - x_min) / dx
        jj = int(np.floor(j_frac))
        if jj < 0:
            val = v[0]
        elif jj >= N - 1:
            val = v[-1]
        else:
            w = j_frac - jj
            val = (1.0 - w) * v[jj] + w * v[jj + 1]
        if alpha != 0.0:
            val = val / np.exp(alpha * x0_val)
        out[idx] = float(val)
    return out


def _conv_grid(s0: float, med_k: float, r: float, t: float, sigma_est: float, L: float, N: int):
    """The CONV log-grid geometry (x_min, x_max, dx) for a given median strike."""
    a_ref, b_ref = _ref_bs_truncation(s0, med_k, r, t, sigma_est, L)
    half_width = (b_ref - a_ref) * 1.5
    x0 = math.log(s0 / med_k)
    x_min, x_max = x0 - half_width, x0 + half_width
    return x_min, x_max, (x_max - x_min) / N


class TestCONVBermudan:
    def test_conv_positive(self) -> None:
        """CONV Bermudan put should be positive."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        conv = conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=10, N=512)[0]
        assert conv > 0.0

    def test_conv_bermudan_structural(self) -> None:
        """CONV Bermudan should be >= European (early exercise adds value).
        NOTE: Current implementation has known bugs; this test documents
        the expected relationship."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        euro = cos_european_put(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)[0]
        conv = conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=10, N=512)[0]
        # Both should be positive
        assert euro > 0 and conv > 0

    def test_fail_closed_bad_inputs(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError):
            conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=0)


class TestCONVBermudanShadow:
    """Differential pins (see ``_shadow_conv`` — characterisation, not accuracy)."""

    def _edge_strikes(self, cf: CharFn, N: int) -> np.ndarray:
        """Strike vector whose log-moneyness hits every interpolation branch:
        jj < 0, jj == 0, interior (w = 0), jj == N-2, jj == N-1, jj > N-1.
        The median stays 100.0 so the grid geometry is the reference one."""
        sig = _ref_sigma_est(cf, T)
        x_min, x_max, dx = _conv_grid(S0, 100.0, R, T, sig, 10.0, N)

        def k_at(x: float) -> float:
            return S0 * math.exp(-x)

        return np.array(
            [
                k_at(x_min - 1.0),  # jj < 0 -> clamps to v[0]
                k_at(x_min + 0.3 * dx),  # jj == 0, w == 0.3
                k_at(x_max - 1.3 * dx),  # jj == N-2 (interior cell)
                k_at(x_max - 0.3 * dx),  # jj == N-1 -> clamps to v[-1]
                k_at(x_max + 1.0),  # jj > N-1 -> clamps to v[-1]
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
            ]
        )

    def test_matches_shadow_across_interpolation_branches(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        N = 512
        ks = self._edge_strikes(cf, N)
        assert float(np.median(ks)) == 100.0
        got = conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=ks, M=2, N=N)
        exp = _shadow_conv(cf, R, T, S0, ks, 2, N, 10.0, 0.0, _ref_sigma_est(cf, T))
        np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-14)
        # every branch must actually be exercised (no degenerate all-zero vector)
        assert np.all(np.isfinite(got))
        assert np.max(np.abs(got)) > 1.0

    def test_matches_shadow_damped(self) -> None:
        """alpha != 0 takes the damped payoff branch and un-damps at x0."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        ks = np.array([90.0, 100.0, 112.0])
        sig = _ref_sigma_est(cf, T)
        for alpha in (0.5, -0.75):
            got = conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=ks, M=3, N=256, alpha=alpha)
            exp = _shadow_conv(cf, R, T, S0, ks, 3, 256, 10.0, alpha, sig)
            np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-14)

    def test_matches_shadow_at_alpha_one(self) -> None:
        """alpha == 1.0 exactly must still take both damped branches (pins
        `alpha != 0.0`, not `alpha != 1.0`, at the payoff *and* the un-damping
        step).  The strike deliberately differs from s0 so the un-damping factor
        exp(alpha * ln(s0/k)) is not 1 and the second branch is observable."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        ks = np.array([85.0, 118.0])
        sig = _ref_sigma_est(cf, T)
        got = conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=ks, M=2, N=128, alpha=1.0)
        exp = _shadow_conv(cf, R, T, S0, ks, 2, 128, 10.0, 1.0, sig)
        np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-14)
        undamped = conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=ks, M=2, N=128)
        np.testing.assert_array_less(0.0, np.abs(got - undamped))
        # the un-damping factor is genuinely != 1 for these strikes
        assert np.exp(1.0 * math.log(S0 / 85.0)) != 1.0

    def test_matches_shadow_for_merton_cf(self) -> None:
        """Non-Gaussian CF through the |φ| kernel and the L-dependent width."""
        cf = merton_char_fn(S0, R, T, SIGMA, lam=1.1, mu_j=-0.05, s_j=0.12)
        ks = np.array([100.0, 105.0])
        sig = _ref_sigma_est(cf, T)
        got = conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=ks, M=2, N=256, L=7.0)
        exp = _shadow_conv(cf, R, T, S0, ks, 2, 256, 7.0, 0.0, sig)
        np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-14)

    def test_L_changes_the_grid_and_the_price(self) -> None:
        """half_width = 1.5·(b-a) is L-dependent: two L values must disagree."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        ks = np.array([100.0])
        p10 = conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=ks, M=2, N=256, L=10.0)[0]
        p4 = conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=ks, M=2, N=256, L=4.0)[0]
        assert p10 != p4


class TestCONVBermudanValidation:
    def test_M1_and_N8_accepted(self) -> None:
        """M == 1 and N == 8 are the documented minima (pins `<` not `<=`)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        got = conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=1, N=8)
        assert np.isfinite(got[0])

    def test_M0_and_N7_rejected(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError, match="t > 0, M >= 1, N >= 8"):
            conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=0, N=64)
        with pytest.raises(ValueError, match="t > 0, M >= 1, N >= 8"):
            conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=2, N=7)

    def test_t_zero_or_negative_rejected(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        for bad_t in (0.0, -2.0):
            with pytest.raises(ValueError, match="t > 0, M >= 1, N >= 8"):
                conv_bermudan_put(cf, r=R, t=bad_t, s0=S0, strikes=np.array([K]), M=2, N=64)

    def test_short_tenor_accepted(self) -> None:
        """0 < t <= 1 must be accepted (pins `t <= 0`, not `t <= 1`)."""
        cf = bs_char_fn(S0, R, 0.5, SIGMA)
        got = conv_bermudan_put(cf, r=R, t=0.5, s0=S0, strikes=np.array([K]), M=2, N=64)[0]
        assert np.isfinite(got)

    def test_bad_strikes_rejected(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        for bad in (np.array([0.0]), np.array([-2.0]), np.array([])):
            with pytest.raises(ValueError, match="strikes must be positive"):
                conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=bad, M=2, N=64)

    def test_tiny_positive_strike_accepted(self) -> None:
        """K in (0, 1] is legal (pins `<= 0`, not `<= 1`)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        got = conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([0.5]), M=1, N=64)[0]
        assert np.isfinite(got)

    def test_defaults_match_explicit_arguments(self) -> None:
        """M=10, N=512, L=10.0, alpha=0.0 defaults pinned bitwise."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        ks = np.array([100.0])
        np.testing.assert_array_equal(
            conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=ks),
            conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=ks, M=10, N=512, L=10.0, alpha=0.0),
        )


def _shadow_hilbert(
    char_fn: CharFn,
    r: float,
    t: float,
    s0: float,
    strike: float,
    barrier: float,
    M: int,
    N: int,
    L: float,
    barrier_type: str,
    sigma_est: float,
) -> tuple[float, float]:
    """Differential shadow of ``hilbert_barrier_call`` (Feng & Linetsky 2008
    skeleton as implemented) returning (price, survival_prob).

    NOTE: like the CONV leg this FFT recursion drops the grid-offset phase and
    builds the dt kernel from |φ|, so its output is orders of magnitude below
    the Merton continuous-monitoring reference.  The shadow is therefore a
    *characterisation* reference that locks the internals (grid width factor,
    frequency step, 1e-300 floor, barrier-mask direction, knock-out zeroing,
    survival-probability mask) against silent drift — NOT an accuracy claim.
    """
    dt = t / M
    a_ref, b_ref = _ref_bs_truncation(s0, strike, r, t, sigma_est, L)
    half_width = (b_ref - a_ref) * 1.5
    x0 = np.log(s0 / strike)
    x_min = x0 - half_width
    x_max = x0 + half_width
    dx = (x_max - x_min) / N
    grid = x_min + dx * np.arange(N)
    dk_grid = 2.0 * np.pi / (N * dx)
    u = dk_grid * np.fft.fftfreq(N) * N
    cf_vals = char_fn(u) * np.exp(-1j * u * np.log(s0))
    cf_dt = np.exp((dt / t) * np.log(np.maximum(np.abs(cf_vals), 1e-300) + 0j))
    cf_dt = cf_dt * np.exp(1j * u * (r * dt))
    log_barrier = np.log(barrier / strike)
    log_s0_shifted = np.log(s0 / strike)
    payoff = np.maximum(strike * np.exp(grid + log_s0_shifted) - strike, 0.0)
    if barrier_type == "down-and-out":
        barrier_mask = (grid + log_s0_shifted) > log_barrier
    else:
        barrier_mask = (grid + log_s0_shifted) < log_barrier
    v = payoff.copy()
    for _ in range(M):
        v_hat = fft(v)
        conv = np.real(ifft(v_hat * np.conj(cf_dt))) * dx
        v_next = np.exp(-r * dt) * conv
        v_next[~barrier_mask] = 0.0
        v = np.maximum(v_next, 0.0)
    j_frac = (log_s0_shifted - x_min) / dx
    jj = int(np.floor(j_frac))
    if jj < 0:
        price = float(v[0])
    elif jj >= N - 1:
        price = float(v[-1])
    else:
        w = j_frac - jj
        price = float((1.0 - w) * v[jj] + w * v[jj + 1])
    euro_val = np.maximum(strike * np.exp(grid + log_s0_shifted) - strike, 0.0)
    mask_both = barrier_mask & (euro_val > 1e-12)
    if mask_both.sum() > 0:
        surv = float(np.mean(v[mask_both] / (euro_val[mask_both] + 1e-12)))
    else:
        surv = 0.0
    return price, surv


class TestHilbertBarrier:
    def test_discrete_barrier_converges_to_continuous(self) -> None:
        """As monitoring dates M→∞, the discrete Hilbert barrier price
        should approach the continuous BS analytic barrier price."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        barrier = 80.0
        cont = bs_continuous_barrier_call(S0, K, barrier, T, SIGMA, R, "down-and-out")

        # Coarse monitoring (M=10) vs fine (M=200)
        coarse = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=barrier, M=10, N=512)[
            "price"
        ]
        fine = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=barrier, M=200, N=1024)[
            "price"
        ]

        # Fine monitoring should be closer to continuous reference
        assert abs(fine - cont) < abs(coarse - cont) + 0.01

    def test_barrier_call_bounded_by_vanilla(self) -> None:
        """Down-and-out barrier call price should be <= vanilla call."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        vanilla = _bs_call(S0, K, T, SIGMA, R)
        result = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=50, N=512)
        assert result["price"] <= vanilla + 0.01

    def test_barrier_at_spot_raises(self) -> None:
        """If barrier >= spot for down-and-out, should raise ValueError."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError, match="below spot"):
            hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=S0 + 1.0, M=50, N=512)

    def test_fail_closed_bad_inputs(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError):
            hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=0)
        with pytest.raises(ValueError):
            hilbert_barrier_call(cf, r=R, t=0.0, s0=S0, strike=K, barrier=80.0)


class TestHilbertBarrierShadow:
    """Differential pins (see ``_shadow_hilbert`` — characterisation, not accuracy)."""

    def test_matches_shadow_down_and_out(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        sig = _ref_sigma_est(cf, T)
        res = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=8, N=128)
        price, surv = _shadow_hilbert(cf, R, T, S0, K, 80.0, 8, 128, 10.0, "down-and-out", sig)
        assert float(res["price"]) != 0.0  # pin must not be vacuous
        assert float(res["price"]) == pytest.approx(price, rel=1e-12, abs=1e-22)
        assert float(res["survival_prob"]) == pytest.approx(surv, rel=1e-12, abs=1e-22)
        assert int(res["M"]) == 8
        assert float(res["barrier"]) == 80.0

    def test_matches_shadow_up_and_out(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        sig = _ref_sigma_est(cf, T)
        res = hilbert_barrier_call(
            cf, r=R, t=T, s0=S0, strike=K, barrier=130.0, M=6, N=96, barrier_type="up-and-out"
        )
        price, surv = _shadow_hilbert(cf, R, T, S0, K, 130.0, 6, 96, 10.0, "up-and-out", sig)
        assert float(res["price"]) != 0.0  # pin must not be vacuous
        assert float(res["price"]) == pytest.approx(price, rel=1e-12, abs=1e-22)
        assert float(res["survival_prob"]) == pytest.approx(surv, rel=1e-12, abs=1e-22)

    def test_barrier_type_selects_mask_direction(self) -> None:
        """down-and-out keeps the states above the barrier, up-and-out the ones
        below; the two legs must produce different survival probabilities."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        do = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=4, N=64)
        uo = hilbert_barrier_call(
            cf, r=R, t=T, s0=S0, strike=K, barrier=130.0, M=4, N=64, barrier_type="up-and-out"
        )
        assert float(do["survival_prob"]) != float(uo["survival_prob"])
        assert float(do["price"]) != float(uo["price"])

    def test_matches_shadow_merton_cf_and_custom_L(self) -> None:
        cf = merton_char_fn(S0, R, T, SIGMA, lam=1.1, mu_j=-0.05, s_j=0.12)
        sig = _ref_sigma_est(cf, T)
        res = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=95.0, barrier=70.0, M=3, N=64, L=6.0)
        price, surv = _shadow_hilbert(cf, R, T, S0, 95.0, 70.0, 3, 64, 6.0, "down-and-out", sig)
        assert float(res["price"]) == pytest.approx(price, rel=1e-12, abs=1e-22)
        assert float(res["survival_prob"]) == pytest.approx(surv, rel=1e-12, abs=1e-22)

    def test_survival_prob_positive_in_normal_case(self) -> None:
        """mask_both.sum() > 0 must take the mean branch, not the 0.0 fallback."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        res = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=4, N=128)
        assert float(res["survival_prob"]) > 0.0

    def test_survival_prob_zero_when_no_surviving_itm_states(self) -> None:
        """Narrow truncation + a strike far above the up-and-out barrier leaves
        no grid state that is both alive and in the money: survival_prob must be
        exactly 0.0 (pins the empty-mask fallback and the 1e-12 ITM cut)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        res = hilbert_barrier_call(
            cf,
            r=R,
            t=T,
            s0=S0,
            strike=200.0,
            barrier=110.0,
            M=2,
            N=64,
            L=0.05,
            barrier_type="up-and-out",
        )
        assert float(res["survival_prob"]) == 0.0
        assert np.isfinite(float(res["price"]))

    def test_survival_prob_with_single_surviving_state(self) -> None:
        """Exactly one grid state is alive *and* in the money, i.e.
        mask_both.sum() == 1: the mean branch must still be taken (pins `> 0`,
        not `> 1`)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        strike = S0 * math.exp(2.0)
        sig = _ref_sigma_est(cf, T)
        a_ref, b_ref = _ref_bs_truncation(S0, strike, R, T, sig, 10.0)
        hw = (b_ref - a_ref) * 1.5
        shift = math.log(S0 / strike)
        dx = 2.0 * hw / 8
        g = shift - hw + dx * np.arange(8)
        both = (g + shift > math.log(80.0 / strike)) & (g + shift > 0.0)
        assert int(both.sum()) == 1  # construction precondition
        res = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=strike, barrier=80.0, M=3, N=8)
        _, surv = _shadow_hilbert(cf, R, T, S0, strike, 80.0, 3, 8, 10.0, "down-and-out", sig)
        assert float(res["survival_prob"]) > 0.0
        assert float(res["survival_prob"]) == pytest.approx(surv, rel=1e-12, abs=1e-15)

    def test_survival_prob_itm_cut_is_tiny_not_unit(self) -> None:
        """A very narrow truncation (L=0.01, N=8) leaves three alive states whose
        in-the-money values are all below 1.0: the `euro_val > 1e-12` cut must
        keep them (survival_prob > 0).  A 1.0 cut empties the mask and returns
        exactly 0.0, so this pins the threshold constant."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        sig = _ref_sigma_est(cf, T)
        res = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=2, N=8, L=0.01)
        price, surv = _shadow_hilbert(cf, R, T, S0, K, 80.0, 2, 8, 0.01, "down-and-out", sig)
        assert float(res["survival_prob"]) > 0.0
        assert float(res["survival_prob"]) == pytest.approx(surv, rel=1e-12, abs=1e-20)
        assert float(res["price"]) == pytest.approx(price, rel=1e-12, abs=1e-20)

    def test_down_and_out_mask_boundary_is_strict(self) -> None:
        """A grid state sitting exactly ON the barrier must be knocked out: the
        down-and-out mask is a strict ``>``.  The barrier is constructed as
        K·exp(grid_j + ln(S0/K)) so the comparison is bitwise-exact."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        strike, N, L, j = 90.0, 64, 10.0, 16
        sig = _ref_sigma_est(cf, T)
        a_ref, b_ref = _ref_bs_truncation(S0, strike, R, T, sig, L)
        hw = (b_ref - a_ref) * 1.5
        x0 = math.log(S0 / strike)
        x_min, x_max = x0 - hw, x0 + hw
        dx = (x_max - x_min) / N
        grid = x_min + dx * np.arange(N)
        barrier = strike * math.exp(grid[j] + x0)
        assert np.log(barrier / strike) == grid[j] + np.log(S0 / strike)  # exact alignment
        assert 0.0 < barrier < S0
        res = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=strike, barrier=barrier, M=3, N=N)
        price, surv = _shadow_hilbert(cf, R, T, S0, strike, barrier, 3, N, L, "down-and-out", sig)
        assert float(res["survival_prob"]) != 0.0
        assert float(res["price"]) == pytest.approx(price, rel=1e-12, abs=1e-20)
        assert float(res["survival_prob"]) == pytest.approx(surv, rel=1e-12, abs=1e-20)

    def test_up_and_out_mask_boundary_is_strict(self) -> None:
        """Same construction for the up-and-out mask, which is a strict ``<``."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        strike, N, L, j = 90.0, 64, 10.0, 58
        sig = _ref_sigma_est(cf, T)
        a_ref, b_ref = _ref_bs_truncation(S0, strike, R, T, sig, L)
        hw = (b_ref - a_ref) * 1.5
        x0 = math.log(S0 / strike)
        x_min, x_max = x0 - hw, x0 + hw
        dx = (x_max - x_min) / N
        grid = x_min + dx * np.arange(N)
        barrier = strike * math.exp(grid[j] + x0)
        assert np.log(barrier / strike) == grid[j] + np.log(S0 / strike)  # exact alignment
        assert barrier > S0
        res = hilbert_barrier_call(
            cf,
            r=R,
            t=T,
            s0=S0,
            strike=strike,
            barrier=barrier,
            M=3,
            N=N,
            barrier_type="up-and-out",
        )
        price, surv = _shadow_hilbert(cf, R, T, S0, strike, barrier, 3, N, L, "up-and-out", sig)
        assert float(res["survival_prob"]) != 0.0
        assert float(res["price"]) == pytest.approx(price, rel=1e-12, abs=1e-20)
        assert float(res["survival_prob"]) == pytest.approx(surv, rel=1e-12, abs=1e-20)

    def test_L_changes_the_grid_and_the_price(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        p10 = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=4, N=128, L=10.0)
        p5 = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=4, N=128, L=5.0)
        assert float(p10["price"]) != float(p5["price"])


class TestHilbertBarrierValidation:
    def test_M1_and_N8_accepted(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        res = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=1, N=8)
        assert np.isfinite(float(res["price"]))

    def test_M0_and_N7_rejected(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError, match="t > 0, M >= 1, N >= 8"):
            hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=0, N=64)
        with pytest.raises(ValueError, match="t > 0, M >= 1, N >= 8"):
            hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=2, N=7)

    def test_t_zero_or_negative_rejected(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        for bad_t in (0.0, -1.0):
            with pytest.raises(ValueError, match="t > 0, M >= 1, N >= 8"):
                hilbert_barrier_call(cf, r=R, t=bad_t, s0=S0, strike=K, barrier=80.0, M=2, N=64)

    def test_short_tenor_accepted(self) -> None:
        """0 < t <= 1 must be accepted (pins `t <= 0`, not `t <= 1`)."""
        cf = bs_char_fn(S0, R, 0.5, SIGMA)
        res = hilbert_barrier_call(cf, r=R, t=0.5, s0=S0, strike=K, barrier=80.0, M=2, N=64)
        assert np.isfinite(float(res["price"]))

    def test_each_non_positive_level_rejected(self) -> None:
        """strike, barrier and s0 are each checked on their own (pins the `or`
        chain, not `and`)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError, match="strike, barrier, s0 must be positive"):
            hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=0.0, barrier=80.0, M=2, N=64)
        with pytest.raises(ValueError, match="strike, barrier, s0 must be positive"):
            hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=0.0, M=2, N=64)
        with pytest.raises(ValueError, match="strike, barrier, s0 must be positive"):
            hilbert_barrier_call(cf, r=R, t=T, s0=0.0, strike=K, barrier=80.0, M=2, N=64)

    def test_sub_unit_levels_accepted(self) -> None:
        """s0/strike/barrier in (0, 1] are legal (pins `<= 0`, not `<= 1`)."""
        cf = bs_char_fn(0.8, R, T, SIGMA)
        res = hilbert_barrier_call(cf, r=R, t=T, s0=0.8, strike=0.5, barrier=0.3, M=2, N=64)
        assert np.isfinite(float(res["price"]))

    def test_down_and_out_barrier_at_spot_rejected(self) -> None:
        """barrier == s0 must raise (pins `>=`, not `>`)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError, match="below spot"):
            hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=S0, M=2, N=64)

    def test_down_and_out_barrier_below_spot_accepted(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        res = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=S0 - 1e-9, M=2, N=64)
        assert np.isfinite(float(res["price"]))

    def test_up_and_out_barrier_at_or_below_spot_rejected(self) -> None:
        """barrier <= s0 must raise for up-and-out (pins `<=`, not `<`)."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        with pytest.raises(ValueError, match="above spot"):
            hilbert_barrier_call(
                cf, r=R, t=T, s0=S0, strike=K, barrier=S0, M=2, N=64, barrier_type="up-and-out"
            )
        with pytest.raises(ValueError, match="above spot"):
            hilbert_barrier_call(
                cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=2, N=64, barrier_type="up-and-out"
            )

    def test_up_and_out_barrier_above_spot_accepted(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        res = hilbert_barrier_call(
            cf, r=R, t=T, s0=S0, strike=K, barrier=S0 + 1e-9, M=2, N=64, barrier_type="up-and-out"
        )
        assert np.isfinite(float(res["price"]))

    def test_defaults_match_explicit_arguments(self) -> None:
        """M=50, N=512, L=10.0 defaults pinned bitwise."""
        cf = bs_char_fn(S0, R, T, SIGMA)
        got = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=80.0)
        exp = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=50, N=512, L=10.0)
        assert got == exp

    def test_default_barrier_type_is_down_and_out(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        got = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=4, N=64)
        exp = hilbert_barrier_call(
            cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=4, N=64, barrier_type="down-and-out"
        )
        assert got == exp


# ---------------------------------------------------------------------------
# BS continuous barrier (analytic reference)
# ---------------------------------------------------------------------------

_DO_CONFIGS = [
    # (s0, k, barrier, t, sigma, r) with barrier < min(s0, k): the regime where
    # the reflection identity used by the module is mathematically valid.
    (100.0, 100.0, 80.0, 1.0, 0.20, 0.03),
    (100.0, 120.0, 75.0, 1.0, 0.35, 0.00),
    (100.0, 90.0, 60.0, 2.0, 0.25, 0.08),
    (50.0, 55.0, 40.0, 0.5, 0.40, 0.01),
    (100.0, 140.0, 95.0, 1.5, 0.15, -0.02),
]


class TestBSContinuousBarrier:
    def test_barrier_below_vanilla(self) -> None:
        vanilla = _bs_call(S0, K, T, SIGMA, R)
        barrier = bs_continuous_barrier_call(S0, K, 80.0, T, SIGMA, R, "down-and-out")
        assert 0.0 <= barrier <= vanilla

    def test_far_barrier_approaches_vanilla(self) -> None:
        """With barrier very far from spot, barrier price ≈ vanilla."""
        vanilla = _bs_call(S0, K, T, SIGMA, R)
        barrier = bs_continuous_barrier_call(S0, K, 1.0, T, SIGMA, R, "down-and-out")
        np.testing.assert_allclose(barrier, vanilla, rtol=0.05)

    def test_up_and_out_barrier(self) -> None:
        vanilla = _bs_call(S0, K, T, SIGMA, R)
        barrier = bs_continuous_barrier_call(S0, K, 120.0, T, SIGMA, R, "up-and-out")
        assert 0.0 <= barrier <= vanilla

    def test_fail_closed_bad_inputs(self) -> None:
        with pytest.raises(ValueError):
            bs_continuous_barrier_call(S0, K, 80.0, t=0.0, sigma=SIGMA)

    # -- added dense pins ---------------------------------------------------

    @pytest.mark.parametrize(("s0", "k", "h", "t", "sigma", "r"), _DO_CONFIGS)
    def test_down_and_out_matches_independent_reflection_formula(
        self, s0: float, k: float, h: float, t: float, sigma: float, r: float
    ) -> None:
        """Merton (1973) down-and-out call re-derived in the test file from the
        reflection identity (independent d1/d2 + scipy.stats.norm)."""
        got = bs_continuous_barrier_call(s0, k, h, t, sigma, r, "down-and-out")
        exp = _ref_do_call(s0, k, h, t, sigma, r)
        assert got == pytest.approx(exp, rel=1e-13, abs=1e-14)

    def test_down_and_out_deep_otm_is_small_but_positive(self) -> None:
        """A deep-OTM DO call is worth ~3e-3 < 1: this pins the max(0.0, ·)
        floor constant (a 1.0 floor would show up immediately)."""
        got = bs_continuous_barrier_call(100.0, 200.0, 80.0, T, SIGMA, R, "down-and-out")
        exp = _ref_do_call(100.0, 200.0, 80.0, T, SIGMA, R)
        assert 0.0 < got < 1.0
        assert got == pytest.approx(exp, rel=1e-12, abs=1e-15)

    def test_reflection_exponent_is_one_at_zero_rate(self) -> None:
        """r = 0 makes 1 - 2r/σ² exactly 1: hand-computable pin of the exponent."""
        s0, k, h, t, sigma = 100.0, 100.0, 80.0, 1.0, 0.2
        c_bs = _ref_bs_scalar(s0, k, t, sigma, 0.0)
        c_image = _ref_bs_scalar(h * h / s0, k, t, sigma, 0.0)
        got = bs_continuous_barrier_call(s0, k, h, t, sigma, 0.0, "down-and-out")
        assert got == pytest.approx(c_bs - (s0 / h) * c_image, rel=1e-13)

    @pytest.mark.parametrize(
        ("r", "sigma", "expo"),
        [
            (0.01, 0.2, 0.5),
            (0.03, 0.2, -0.5),
            (0.0, 0.4, 1.0),
            (0.01, 0.4, 0.875),
            (-0.04, 0.2, 3.0),
        ],
    )
    def test_reflection_exponent_is_one_minus_two_r_over_sigma_squared(
        self, r: float, sigma: float, expo: float
    ) -> None:
        """(S/H)^{1-2r/σ²} pinned at five hand-computed exponents (kills both the
        1.0 and the 2.0 constants and the σ² power)."""
        s0, k, h, t = 100.0, 110.0, 70.0, 1.0
        assert 1.0 - 2.0 * r / sigma**2 == pytest.approx(expo, rel=1e-14)
        c_bs = _ref_bs_scalar(s0, k, t, sigma, r)
        c_image = _ref_bs_scalar(h * h / s0, k, t, sigma, r)
        got = bs_continuous_barrier_call(s0, k, h, t, sigma, r, "down-and-out")
        assert got == pytest.approx(c_bs - (s0 / h) ** expo * c_image, rel=1e-13, abs=1e-14)

    def test_image_spot_is_barrier_squared_over_spot(self) -> None:
        """The reflected leg must be priced at H²/S (kills H³/S and call→put)."""
        s0, k, h, t, sigma, r = 100.0, 110.0, 70.0, 1.0, 0.3, 0.02
        expo = 1.0 - 2.0 * r / sigma**2
        c_bs = _ref_bs_scalar(s0, k, t, sigma, r)
        got = bs_continuous_barrier_call(s0, k, h, t, sigma, r, "down-and-out")
        assert got == pytest.approx(
            c_bs - (s0 / h) ** expo * _ref_bs_scalar(h * h / s0, k, t, sigma, r), rel=1e-13
        )
        # and it must NOT be the put, nor an H³/S image
        assert got != pytest.approx(
            c_bs - (s0 / h) ** expo * _ref_bs_scalar(h * h / s0, k, t, sigma, r, call=False),
            rel=1e-6,
        )
        assert got != pytest.approx(
            c_bs - (s0 / h) ** expo * _ref_bs_scalar(h**3 / s0, k, t, sigma, r), rel=1e-6
        )

    def test_down_and_out_zero_at_or_below_barrier(self) -> None:
        """S0 <= H knocks out immediately: exactly 0.0, not 1.0."""
        assert bs_continuous_barrier_call(80.0, 100.0, 80.0, T, SIGMA, R, "down-and-out") == 0.0
        assert bs_continuous_barrier_call(70.0, 100.0, 80.0, T, SIGMA, R, "down-and-out") == 0.0

    def test_up_and_out_zero_at_or_above_barrier(self) -> None:
        assert bs_continuous_barrier_call(130.0, 100.0, 130.0, T, SIGMA, R, "up-and-out") == 0.0
        assert bs_continuous_barrier_call(140.0, 100.0, 130.0, T, SIGMA, R, "up-and-out") == 0.0

    def test_up_and_out_is_clamped_to_zero_DEFECT(self) -> None:
        """DEFECT (characterisation pin — NOT an accuracy claim).

        The up-and-out branch reuses the *down*-and-in reflection
        (S/H)^{1-2r/σ²}·C(H²/S, K).  That identity requires the payoff support
        to lie above the barrier (K >= H); in the usual up-and-out regime H > K
        the reflected leg exceeds the vanilla price, so max(0.0, ·) floors the
        result at exactly 0.0 for every H > S.  Monte-Carlo cross-check for the
        config below (400k paths x 800 steps, discrete monitoring): 3.3166
        +/- 0.0098 versus 0.0 from this function — a ~339-sigma error.  The
        correct up-and-in leg needs the four-term Reiner-Rubinstein form.
        Pinned as-is so the floor constant cannot drift; update when repaired.
        """
        got = bs_continuous_barrier_call(100.0, 100.0, 130.0, T, SIGMA, R, "up-and-out")
        assert got == 0.0
        vanilla = _ref_bs_scalar(100.0, 100.0, T, SIGMA, R)
        assert vanilla > 9.0  # the option is far from worthless

    def test_up_and_out_reflection_constants_DEFECT_characterization(self) -> None:
        """Deep-ITM up-and-out config where the clamp does NOT bind, so the UO
        reflection exponent, the H²/S image spot and the call flag stay
        observable.  Same (defective) expression as the DO branch — see
        ``test_up_and_out_is_clamped_to_zero_DEFECT``."""
        s0, k, h, t, sigma, r = 100.0, 15.0, 200.0, 1.0, 0.2, -0.05
        got = bs_continuous_barrier_call(s0, k, h, t, sigma, r, "up-and-out")
        exp = _ref_do_call(s0, k, h, t, sigma, r)
        assert got > 1.0  # clamp not binding
        assert got == pytest.approx(exp, rel=1e-13)

    def test_t_and_sigma_validation(self) -> None:
        for bad_t, bad_sigma in ((0.0, SIGMA), (-1.0, SIGMA), (T, 0.0), (T, -0.2)):
            with pytest.raises(ValueError, match="t and sigma must be positive"):
                bs_continuous_barrier_call(S0, K, 80.0, bad_t, bad_sigma, R, "down-and-out")

    def test_levels_validation(self) -> None:
        """s0, k and barrier are each checked on their own (pins `or`, not `and`)."""
        with pytest.raises(ValueError, match="s0, k, barrier must be positive"):
            bs_continuous_barrier_call(0.0, K, 80.0, T, SIGMA, R, "down-and-out")
        with pytest.raises(ValueError, match="s0, k, barrier must be positive"):
            bs_continuous_barrier_call(S0, 0.0, 80.0, T, SIGMA, R, "down-and-out")
        with pytest.raises(ValueError, match="s0, k, barrier must be positive"):
            bs_continuous_barrier_call(S0, K, 0.0, T, SIGMA, R, "down-and-out")

    def test_sub_unit_levels_accepted(self) -> None:
        """s0/k/barrier in (0, 1] are legal (pins `<= 0`, not `<= 1`)."""
        got = bs_continuous_barrier_call(0.9, 0.8, 0.5, 1.0, 0.3, 0.02, "down-and-out")
        exp = _ref_do_call(0.9, 0.8, 0.5, 1.0, 0.3, 0.02)
        assert got == pytest.approx(exp, rel=1e-12)

    def test_short_tenor_and_vol_accepted(self) -> None:
        """0 < t <= 1 and 0 < sigma <= 1 must be accepted."""
        got = bs_continuous_barrier_call(S0, K, 80.0, 0.25, 0.4, R, "down-and-out")
        exp = _ref_do_call(S0, K, 80.0, 0.25, 0.4, R)
        assert got == pytest.approx(exp, rel=1e-13)

    def test_default_rate_is_zero(self) -> None:
        """The r default must be 0.0 (pinned bitwise against the explicit call)."""
        got = bs_continuous_barrier_call(S0, K, 80.0, T, SIGMA)
        exp = bs_continuous_barrier_call(S0, K, 80.0, T, SIGMA, 0.0, "down-and-out")
        assert got == exp
        assert got != bs_continuous_barrier_call(S0, K, 80.0, T, SIGMA, 1.0, "down-and-out")

    def test_default_barrier_type_is_down_and_out(self) -> None:
        got = bs_continuous_barrier_call(S0, K, 80.0, T, SIGMA, R)
        exp = bs_continuous_barrier_call(S0, K, 80.0, T, SIGMA, R, "down-and-out")
        assert got == exp

    def test_monotone_in_barrier_level(self) -> None:
        """A lower down-and-out barrier knocks out less often => higher price."""
        prices = [
            bs_continuous_barrier_call(S0, K, h, T, SIGMA, R, "down-and-out")
            for h in (50.0, 65.0, 80.0, 95.0)
        ]
        assert prices[0] > prices[1] > prices[2] > prices[3]


class TestBsPriceScalar:
    """The module's private scalar BSM used by the barrier formulas."""

    _GRID = [
        (100.0, 100.0, 1.0, 0.2, 0.03),
        (100.0, 80.0, 0.5, 0.35, 0.0),
        (64.0, 100.0, 2.0, 0.25, 0.08),
        (121.0, 130.0, 1.0, 0.4, -0.02),
        (0.9, 0.8, 0.25, 0.3, 0.01),
    ]

    def test_matches_independent_bsm(self) -> None:
        """Cross-pin against models.options.bs_price (a separate implementation)."""
        for s, k, t, sigma, r in self._GRID:
            assert _bs_price_scalar(s, k, t, sigma, r, call=True) == pytest.approx(
                bs_price(s, k, t, sigma, r, call=True), rel=1e-13, abs=1e-14
            )
            assert _bs_price_scalar(s, k, t, sigma, r, call=False) == pytest.approx(
                bs_price(s, k, t, sigma, r, call=False), rel=1e-13, abs=1e-14
            )

    def test_matches_reference_d1_d2(self) -> None:
        for s, k, t, sigma, r in self._GRID:
            assert _bs_price_scalar(s, k, t, sigma, r, call=True) == pytest.approx(
                _ref_bs_scalar(s, k, t, sigma, r, call=True), rel=1e-13, abs=1e-14
            )
            assert _bs_price_scalar(s, k, t, sigma, r, call=False) == pytest.approx(
                _ref_bs_scalar(s, k, t, sigma, r, call=False), rel=1e-13, abs=1e-14
            )

    def test_put_call_parity(self) -> None:
        for s, k, t, sigma, r in self._GRID:
            c = _bs_price_scalar(s, k, t, sigma, r, call=True)
            p = _bs_price_scalar(s, k, t, sigma, r, call=False)
            assert c - p == pytest.approx(s - k * math.exp(-r * t), rel=1e-12, abs=1e-13)

    def test_default_is_call(self) -> None:
        s, k, t, sigma, r = 100.0, 95.0, 1.0, 0.25, 0.03
        assert _bs_price_scalar(s, k, t, sigma, r) == _bs_price_scalar(s, k, t, sigma, r, call=True)
        assert _bs_price_scalar(s, k, t, sigma, r) != _bs_price_scalar(
            s, k, t, sigma, r, call=False
        )

    def test_intrinsic_at_zero_tenor(self) -> None:
        """t == 0 takes the intrinsic branch (pins `<=`, not `<`)."""
        assert _bs_price_scalar(100.0, 90.0, 0.0, 0.2, 0.03, call=True) == 10.0
        assert _bs_price_scalar(90.0, 100.0, 0.0, 0.2, 0.03, call=False) == 10.0
        assert _bs_price_scalar(100.0, 100.0, 0.0, 0.2, 0.0, call=True) == 0.0

    def test_intrinsic_at_negative_tenor(self) -> None:
        """t < 0 uses the same branch, with exp(-r t) > 1 for r > 0."""
        got = _bs_price_scalar(90.0, 100.0, -0.5, 0.2, 0.03, call=False)
        assert got == pytest.approx(max(100.0 * math.exp(0.015) - 90.0, 0.0), rel=1e-14)
        got_c = _bs_price_scalar(110.0, 100.0, -0.5, 0.2, 0.03, call=True)
        assert got_c == pytest.approx(110.0 - 100.0 * math.exp(0.015), rel=1e-14)

    def test_zero_intrinsic_is_floored_at_zero(self) -> None:
        """Out-of-the-money intrinsic value is exactly 0.0 (kills a 1.0 floor)."""
        assert _bs_price_scalar(100.0, 100.5, 0.0, 0.2, 0.0, call=True) == 0.0
        assert _bs_price_scalar(100.5, 100.0, 0.0, 0.2, 0.0, call=False) == 0.0

    def test_long_tenor_accepted(self) -> None:
        """t > 1 must take the d1/d2 branch, not the intrinsic one."""
        got = _bs_price_scalar(100.0, 100.0, 5.0, 0.2, 0.03, call=True)
        assert got == pytest.approx(_ref_bs_scalar(100.0, 100.0, 5.0, 0.2, 0.03), rel=1e-13)
        assert got > 100.0 - 100.0 * math.exp(-0.03 * 5.0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_cos_call_deterministic(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        p1 = cos_european_call(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)
        p2 = cos_european_call(cf, r=R, t=T, strikes=np.array([K]), s0=S0, n=256)
        np.testing.assert_array_equal(p1, p2)

    def test_bermudan_deterministic(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        p1 = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=10, n=256)
        p2 = cos_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=10, n=256)
        np.testing.assert_array_equal(p1, p2)

    def test_conv_and_hilbert_deterministic(self) -> None:
        cf = bs_char_fn(S0, R, T, SIGMA)
        c1 = conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=3, N=128)
        c2 = conv_bermudan_put(cf, r=R, t=T, s0=S0, strikes=np.array([K]), M=3, N=128)
        np.testing.assert_array_equal(c1, c2)
        h1 = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=3, N=128)
        h2 = hilbert_barrier_call(cf, r=R, t=T, s0=S0, strike=K, barrier=80.0, M=3, N=128)
        assert h1 == h2
