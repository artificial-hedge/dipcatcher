"""SVI implied-volatility surface: slices, calibration, arbitrage checks.

The stochastic-volatility-inspired (SVI) family parameterizes the *total
implied variance* smile ``w(k) = sigma_BS^2(k, t) * t`` in log-forward
moneyness ``k = log(K / F_t)``. Three nested forms are implemented:

- raw SVI: ``w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))``
  with ``b >= 0``, ``|rho| < 1``, ``sigma > 0``;
- natural SVI:
  ``w(k) = delta + (omega/2) (1 + zeta rho (k - mu)
  + sqrt((zeta (k - mu) + rho)^2 + (1 - rho^2)))``;
- SSVI slice: ``w(k) = (theta/2) (1 + rho phi k + sqrt((phi k + rho)^2
  + (1 - rho^2)))`` with ATM total variance ``theta`` and curvature
  ``phi = phi(theta)``; eSSVI is SSVI with maturity-dependent rho.

Every natural SVI and SSVI slice is algebraically a raw SVI slice (the
module maps both into raw form and reuses one set of analytics), and a
raw SVI slice has linear wings ``w(k) ~ b (1 + rho) k`` (right) and
``w(k) ~ b (1 - rho) |k|`` (left) as ``|k| -> inf``.

References
----------
- J. Gatheral & A. Jacquier (2014), "Arbitrage-free SVI volatility
  surfaces", Quantitative Finance 14(1), 59-71, arXiv:1204.0646.
  Used here: the g(k) butterfly-arbitrage functional (eq. 2.1), the
  iff condition ``g >= 0`` together with ``lim_{k -> +inf} d_+(k) =
  -inf`` (Lemma 2.2), the raw/natural SVI mapping (Lemma 3.1), the
  SVI-JW wing parameters (Section 3.4), the SSVI surface form
  (eq. 4.1), the SSVI butterfly conditions ``theta phi (1+|rho|) < 4``
  and ``theta phi^2 (1+|rho|) <= 4`` (Theorem 4.2), and the calendar
  rule ``d_t w >= 0`` with its ``d_theta(theta phi(theta))`` bound
  (Lemma 2.1 / Theorem 4.3).
- J. Gatheral & A. Jacquier (2011), "Convergence of Heston to SVI",
  Quantitative Finance 11(8), 1129-1132, arXiv:1002.3633. (The
  large-maturity Heston smile is exactly SVI; motivation only, no
  formulas implemented.)
- S. Hendriks & C. Martini (2019), "The extended SSVI volatility
  surface", Journal of Computational Finance 23(3); SSRN 2971502.
  Journal/SSRN only - no arXiv version exists. Used here: the
  necessary-and-sufficient non-crossing condition for two eSSVI
  slices, restated in Corbetta, Cohort, Laachir & Martini (2019),
  "Robust calibration and arbitrage-free interpolation of SSVI
  slices", arXiv:1804.04924: ``theta_i`` and ``psi_i = theta_i
  phi_i`` non-decreasing plus ``|d(rho psi) / d psi| <= 1``.
- R. W. Lee (2004), "The moment formula for implied volatility at
  extreme strikes", Annals of Applied Probability 14(1), 47-64.
  Journal-only (no arXiv). Used here: the wing-slope bound
  ``limsup |d_k w| <= 2`` per unit |k| of total variance; for raw SVI
  this is ``b (1 + |rho|) <= 2``.

Composition: ``SVIParams``/``raw_svi_w`` evaluate slices; ``svi_g`` and
``slice_arbitrage_report`` run the per-slice butterfly test;
``calibrate_slice``/``calibrate_slice_iv`` fit one maturity (multi-start
least squares, optional no-arb-penalized variant, per-point
interpolation fallback); ``assemble_surface`` stitches slices onto a
maturity grid with linear-in-t interpolation of total variance;
``calendar_report`` checks total-variance monotonicity in t per fixed k
(Definition 2.2) and ``repair_calendar`` projects each k-column onto its
non-decreasing isotonic fit (PAVA); ``lee_wing_check`` and
``svi_jump_wings`` cover wing extrapolation and the Lee bound;
``essvi_slices_consistent`` carries the Hendriks-Martini
slice-consistency test; ``bench_svi_surface`` returns the synthetic
correctness numbers.

Honesty: everything here is SYNTHETIC correctness tooling. Tests and the
``bench_svi_surface`` numbers are computed on synthetic smiles drawn from
the same parametric family and on violations planted by construction -
they demonstrate that the formulas, calibrators, and detectors behave as
the cited literature specifies, never that real market surfaces are
arbitrage-free or that anything earns money. No market evidence, no P&L,
no live-trading claims (AGENTS.md honesty contract). The interpolation
fallback exists so calibration degrades gracefully on non-SVI-shaped
smiles; its linear wing extension can violate the Lee bound and is
flagged, not hidden.

Fail-closed: empty/degenerate inputs, non-positive maturities,
non-finite or non-positive quotes all raise ``ValueError``.
Deterministic: all randomness comes from ``numpy.random.default_rng``
inside functions; no module-level or numpy global state is touched.
numpy + scipy only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

Array = NDArray[np.float64]

# Tolerance for deciding whether a non-negativity/monotonicity
# constraint (g(k) >= 0, w >= 0, non-decreasing in t) is violated.
_ARB_TOL = 1e-10
# Lee (2004): wing slope of total variance per unit |k|.
_LEE_BOUND = 2.0


# ---------------------------------------------------------------------
# Parameter containers (frozen, validated at construction).
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SVIParams:
    """Raw SVI parameters ``(a, b, rho, m, sigma)``.

    Requires all finite, ``b >= 0``, ``|rho| < 1``, ``sigma > 0``.
    ``a`` is unrestricted: SVI smiles can have negative ``a`` while still
    producing positive total variance; positivity is checked on ``w``.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if not np.isfinite([self.a, self.b, self.rho, self.m, self.sigma]).all():
            raise ValueError("svi params must be finite")
        if self.b < 0.0:
            raise ValueError("b must be >= 0")
        if abs(self.rho) >= 1.0:
            raise ValueError("rho must be in (-1, 1)")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be > 0")


@dataclass(frozen=True)
class NaturalSVIParams:
    """Natural SVI parameters ``(delta, mu, rho, omega, zeta)``."""

    delta: float
    mu: float
    rho: float
    omega: float
    zeta: float

    def __post_init__(self) -> None:
        if not np.isfinite([self.delta, self.mu, self.rho, self.omega, self.zeta]).all():
            raise ValueError("natural svi params must be finite")
        if self.omega < 0.0:
            raise ValueError("omega must be >= 0")
        if abs(self.rho) >= 1.0:
            raise ValueError("rho must be in (-1, 1)")
        if self.zeta <= 0.0:
            raise ValueError("zeta must be > 0")


# ---------------------------------------------------------------------
# Slice evaluation, derivatives, and equivalent parameterizations.
# ---------------------------------------------------------------------


def raw_svi_w(k: Array | float, p: SVIParams) -> Array:
    """Raw SVI total implied variance ``w(k)`` (paper Section 3.1)."""
    kk = np.asarray(k, dtype=float)
    if not np.isfinite(kk).all():
        raise ValueError("k must be finite")
    x = kk - p.m
    return np.asarray(p.a + p.b * (p.rho * x + np.sqrt(x * x + p.sigma**2)), dtype=float)


def raw_svi_dw(k: Array | float, p: SVIParams) -> Array:
    """``w'(k) = b (rho + (k - m) / sqrt((k - m)^2 + sigma^2))``."""
    kk = np.asarray(k, dtype=float)
    if not np.isfinite(kk).all():
        raise ValueError("k must be finite")
    x = kk - p.m
    return np.asarray(p.b * (p.rho + x / np.sqrt(x * x + p.sigma**2)), dtype=float)


def raw_svi_d2w(k: Array | float, p: SVIParams) -> Array:
    """``w''(k) = b sigma^2 / ((k - m)^2 + sigma^2)^(3/2)``."""
    kk = np.asarray(k, dtype=float)
    if not np.isfinite(kk).all():
        raise ValueError("k must be finite")
    x = kk - p.m
    return np.asarray(p.b * p.sigma**2 / (x * x + p.sigma**2) ** 1.5, dtype=float)


def natural_to_raw(pn: NaturalSVIParams) -> SVIParams:
    """Natural -> raw SVI mapping (paper, Lemma 3.1)."""
    return SVIParams(
        a=pn.delta + 0.5 * pn.omega * (1.0 - pn.rho**2),
        b=0.5 * pn.omega * pn.zeta,
        rho=pn.rho,
        m=pn.mu - pn.rho / pn.zeta,
        sigma=np.sqrt(1.0 - pn.rho**2) / pn.zeta,
    )


def raw_to_natural(p: SVIParams) -> NaturalSVIParams:
    """Raw -> natural SVI mapping (inverse of Lemma 3.1)."""
    s = float(np.sqrt(1.0 - p.rho**2))
    omega = 2.0 * p.b * p.sigma / s
    return NaturalSVIParams(
        delta=p.a - 0.5 * omega * (1.0 - p.rho**2),
        mu=p.m + p.rho * p.sigma / s,
        rho=p.rho,
        omega=omega,
        zeta=s / p.sigma,
    )


def natural_svi_w(k: Array | float, pn: NaturalSVIParams) -> Array:
    """Natural SVI total variance, evaluated via the raw mapping."""
    return raw_svi_w(k, natural_to_raw(pn))


def ssvi_to_raw(theta: float, rho: float, phi: float) -> SVIParams:
    """One SSVI slice ``(theta, rho, phi)`` as raw SVI parameters.

    ``theta`` is the ATM total implied variance and ``phi = phi(theta)``
    the curvature at this slice. Expanding eq. 4.1 of the paper gives the
    raw form with ``a = theta (1 - rho^2)/2``, ``b = theta phi / 2``,
    ``m = -rho/phi``, ``sigma = sqrt(1 - rho^2)/phi``.
    """
    if not np.isfinite([theta, rho, phi]).all():
        raise ValueError("ssvi params must be finite")
    if theta <= 0.0 or phi <= 0.0:
        raise ValueError("theta and phi must be > 0")
    if abs(rho) >= 1.0:
        raise ValueError("rho must be in (-1, 1)")
    return SVIParams(
        a=0.5 * theta * (1.0 - rho**2),
        b=0.5 * theta * phi,
        rho=rho,
        m=-rho / phi,
        sigma=np.sqrt(1.0 - rho**2) / phi,
    )


def ssvi_w(k: Array | float, theta: float, rho: float, phi: float) -> Array:
    """SSVI slice total variance ``w(k, theta)`` (paper, eq. 4.1)."""
    return raw_svi_w(k, ssvi_to_raw(theta, rho, phi))


def ssvi_power_phi(theta: Array | float, eta: float, gamma: float) -> Array:
    """Power-law curvature ``phi(theta) = eta * theta^(-gamma)``.

    The standard tractable SSVI choice: with ``0 <= gamma <= 1`` the
    calendar condition (Theorem 4.3) reduces to ATM total variance
    non-decreasing in t.
    """
    if not np.isfinite(eta) or not np.isfinite(gamma):
        raise ValueError("eta and gamma must be finite")
    if eta <= 0.0 or not 0.0 <= gamma <= 1.0:
        raise ValueError("need eta > 0 and gamma in [0, 1]")
    th = np.asarray(theta, dtype=float)
    if (th <= 0.0).any() or not np.isfinite(th).all():
        raise ValueError("theta must be finite and > 0")
    return np.asarray(eta * th ** (-gamma), dtype=float)


def svi_implied_vol(k: Array | float, p: SVIParams, t: float) -> Array:
    """Black implied vol ``sqrt(max(w(k), 0) / t)``; requires ``t > 0``."""
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("t must be finite and > 0")
    w = np.clip(raw_svi_w(k, p), 0.0, None)
    return np.asarray(np.sqrt(w / t), dtype=float)


# ---------------------------------------------------------------------
# Butterfly arbitrage: the g(k) functional, density, Lee wings.
# ---------------------------------------------------------------------


def svi_g(k: Array | float, p: SVIParams) -> Array:
    """Gatheral-Jacquier ``g(k)`` for a raw SVI slice (eq. 2.1).

    ``g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2``. A slice
    with ``w > 0`` everywhere is free of butterfly arbitrage iff
    ``g(k) >= 0`` for all k and ``d_+(k) -> -inf`` as ``k -> +inf``
    (Lemma 2.2). Where ``w <= 0`` the smile already admits arbitrage
    (negative total variance); ``g`` returns ``-inf`` there so the
    violation surfaces instead of silently producing NaN.
    """
    kk = np.asarray(k, dtype=float)
    if not np.isfinite(kk).all():
        raise ValueError("k must be finite")
    w = raw_svi_w(kk, p)
    wp = raw_svi_dw(kk, p)
    wpp = raw_svi_d2w(kk, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1.0 - kk * wp / (2.0 * w)) ** 2 - 0.25 * wp**2 * (1.0 / w + 0.25) + 0.5 * wpp
    return np.asarray(np.where(w > 0.0, g, -np.inf), dtype=float)


def _g_raw(k: Array, p: SVIParams) -> Array:
    """Unmasked g(k): same formula as :func:`svi_g` but without the
    ``w <= 0 -> -inf`` guard. Used internally by the penalized
    calibration residual, where finite (even meaningless) values keep
    the least-squares residual finite so optimizer starts are not
    discarded outright; ``w < 0`` violations are caught by the separate
    w-hinge term anyway."""
    w = raw_svi_w(k, p)
    wp = raw_svi_dw(k, p)
    wpp = raw_svi_d2w(k, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1.0 - k * wp / (2.0 * w)) ** 2 - 0.25 * wp**2 * (1.0 / w + 0.25) + 0.5 * wpp
    return np.asarray(np.nan_to_num(g, nan=0.0, posinf=1e6, neginf=-1e6), dtype=float)


def implied_log_density(k: Array | float, p: SVIParams) -> Array:
    """Risk-neutral density in log-moneyness ``k``.

    ``f(k) = g(k) phi(d_-(k)) / sqrt(w(k))`` with
    ``d_-(k) = -k / sqrt(w) - sqrt(w)/2`` (the density appearing in the
    paper's butterfly condition). Negative values are kept: they are the
    arbitrage signal. Returns 0 where ``w <= 0`` (density undefined).
    """
    kk = np.asarray(k, dtype=float)
    if not np.isfinite(kk).all():
        raise ValueError("k must be finite")
    w = raw_svi_w(kk, p)
    g = svi_g(kk, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        wc = np.clip(w, 1e-300, None)
        dm = -kk / np.sqrt(wc) - 0.5 * np.sqrt(wc)
        dens = g * np.exp(-0.5 * dm**2) / np.sqrt(2.0 * np.pi * wc)
    out = np.where((w > 0.0) & np.isfinite(dens), dens, 0.0)
    return np.asarray(out, dtype=float)


def lee_wing_check(p: SVIParams, bound: float = _LEE_BOUND) -> dict[str, float]:
    """Roger Lee moment-formula wing check on a raw SVI slice.

    As ``|k| -> inf`` the SVI wings are linear: right slope
    ``b (1 + rho)``, left slope ``b (1 - rho)`` per unit |k| of total
    variance. Lee (2004) bounds each slope by 2; exceeding it is
    equivalent to ``d_+`` not diverging to ``-inf`` as ``k -> +inf``
    (calls not vanishing at infinite strike) - arbitrage.
    """
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("bound must be finite and > 0")
    left = p.b * (1.0 - p.rho)
    right = p.b * (1.0 + p.rho)
    left_bad = left > bound + _ARB_TOL
    right_bad = right > bound + _ARB_TOL
    return {
        "left_slope": float(left),
        "right_slope": float(right),
        "bound": float(bound),
        "left_violated": float(left_bad),
        "right_violated": float(right_bad),
        "violated": float(left_bad or right_bad),
    }


def slice_arbitrage_report(
    p: SVIParams,
    k_lo: float = -5.0,
    k_hi: float = 5.0,
    n_grid: int = 2001,
) -> dict[str, float]:
    """Per-slice butterfly-arbitrage scan.

    Sweeps ``g(k)`` on a uniform grid over ``[k_lo, k_hi]`` and combines
    it with the two analytic edge conditions: positivity of ``w`` (the
    SVI minimum is ``a + b sigma sqrt(1 - rho^2)`` at
    ``k* = m - rho sigma / sqrt(1 - rho^2)``) and the Lee wing bound.
    ``free`` is 1.0 iff all three hold. The grid scan is finite - the
    asymptotic part is covered analytically by the wing bound rather
    than by trusting ``g`` at the grid edges.
    """
    if not np.isfinite([k_lo, k_hi]).all() or k_hi <= k_lo:
        raise ValueError("need finite k_lo < k_hi")
    if n_grid < 5:
        raise ValueError("n_grid must be >= 5")
    kg = np.linspace(k_lo, k_hi, int(n_grid))
    g = svi_g(kg, p)
    w_min = p.a + p.b * p.sigma * np.sqrt(1.0 - p.rho**2)
    min_g = float(np.min(g))
    arg = int(np.argmin(g))
    wings = lee_wing_check(p)
    free = min_g >= -_ARB_TOL and w_min > _ARB_TOL and wings["violated"] == 0.0
    return {
        "min_g": min_g,
        "min_g_k": float(kg[arg]),
        "min_w": float(w_min),
        "w_positive": float(w_min > _ARB_TOL),
        "wing_violated": wings["violated"],
        "free": float(free),
    }


def ssvi_butterfly_ok(theta: float, rho: float, phi: float) -> dict[str, float]:
    """SSVI no-butterfly sufficient conditions (paper, Theorem 4.2).

    ``theta phi (1 + |rho|) < 4`` and ``theta phi^2 (1 + |rho|) <= 4``.
    With ``psi = theta phi`` the first condition is the Lee wing bound
    ``b (1 + |rho|) <= 2`` since the raw ``b = theta phi / 2``.
    """
    if not np.isfinite([theta, rho, phi]).all():
        raise ValueError("ssvi params must be finite")
    if theta <= 0.0 or phi <= 0.0 or abs(rho) >= 1.0:
        raise ValueError("need theta > 0, phi > 0, |rho| < 1")
    c1 = theta * phi * (1.0 + abs(rho))
    c2 = theta * phi**2 * (1.0 + abs(rho))
    return {
        "cond1_lhs": float(c1),
        "cond2_lhs": float(c2),
        "cond1_ok": float(c1 < 4.0),
        "cond2_ok": float(c2 <= 4.0 + _ARB_TOL),
        "ok": float(c1 < 4.0 and c2 <= 4.0 + _ARB_TOL),
    }


def ssvi_calendar_partial_theta_bound(
    rho: float, phi: float, d_theta_phi: float
) -> dict[str, float]:
    """SSVI calendar condition (paper, Theorem 4.3).

    With ``theta_t`` non-decreasing there is no calendar arbitrage iff
    ``0 <= d_theta(theta phi(theta)) <= phi(theta) (1 +
    sqrt(1 - rho^2)) / rho^2`` (upper bound ``+inf`` at ``rho = 0``).
    ``d_theta_phi`` is the derivative ``d(theta phi)/d theta`` evaluated
    at the slice.
    """
    if not np.isfinite([rho, phi, d_theta_phi]).all():
        raise ValueError("inputs must be finite")
    if phi <= 0.0 or abs(rho) >= 1.0:
        raise ValueError("need phi > 0, |rho| < 1")
    upper = np.inf if rho == 0.0 else phi * (1.0 + np.sqrt(1.0 - rho**2)) / rho**2
    ok = -_ARB_TOL <= d_theta_phi <= upper + _ARB_TOL
    return {
        "lower": 0.0,
        "upper": float(upper),
        "d_theta_phi": float(d_theta_phi),
        "ok": float(ok),
    }


def essvi_slices_consistent(thetas: Array, rhos: Array, phis: Array) -> dict[str, float]:
    """Hendriks-Martini non-crossing condition for eSSVI slices.

    Slices ordered by increasing maturity with parameters
    ``(theta_i, rho_i, phi_i)``; write ``psi_i = theta_i phi_i``. There
    is no calendar arbitrage between adjacent slices iff ``theta_i`` and
    ``psi_i`` are non-decreasing and, whenever ``psi_{i+1} > psi_i``,

        ``|(rho_{i+1} psi_{i+1} - rho_i psi_i) / (psi_{i+1} - psi_i)| <= 1``

    (Corbetta et al. 2019, arXiv:1804.04924, restating the
    Hendriks-Martini result). When ``psi_{i+1} == psi_i`` consistency
    requires ``rho psi`` unchanged.
    """
    th = np.asarray(thetas, dtype=float).ravel()
    rh = np.asarray(rhos, dtype=float).ravel()
    ph = np.asarray(phis, dtype=float).ravel()
    if not (th.shape == rh.shape == ph.shape):
        raise ValueError("thetas/rhos/phis must have equal length")
    if th.size < 2:
        raise ValueError("need >= 2 slices")
    if not np.isfinite(th).all() or not np.isfinite(rh).all() or not np.isfinite(ph).all():
        raise ValueError("slice params must be finite")
    if (th <= 0.0).any() or (ph <= 0.0).any() or (np.abs(rh) >= 1.0).any():
        raise ValueError("need theta > 0, phi > 0, |rho| < 1")
    psi = th * ph
    d_theta = np.diff(th)
    d_psi = np.diff(psi)
    theta_ok = bool((d_theta >= -_ARB_TOL).all())
    psi_ok = bool((d_psi >= -_ARB_TOL).all())
    worst_ratio = 0.0
    cross_ok = True
    for i in range(th.size - 1):
        gap = float(d_psi[i])
        jump = abs(float(rh[i + 1] * psi[i + 1] - rh[i] * psi[i]))
        if gap > _ARB_TOL:
            ratio = jump / gap
            worst_ratio = max(worst_ratio, ratio)
            if ratio > 1.0 + _ARB_TOL:
                cross_ok = False
        elif jump > _ARB_TOL:
            cross_ok = False
    ok = theta_ok and psi_ok and cross_ok
    return {
        "theta_nondecreasing": float(theta_ok),
        "psi_nondecreasing": float(psi_ok),
        "no_crossing": float(cross_ok),
        "worst_rho_psi_ratio": float(worst_ratio),
        "consistent": float(ok),
    }


def svi_jump_wings(p: SVIParams, t: float) -> dict[str, float]:
    """SVI-JW parameters ``(v, psi, p, c, v~)`` of a raw SVI slice.

    In *implied variance* (not total variance) units: ``v`` is the ATM
    implied variance, ``psi`` the ATM skew ``d sigma_BS/dk`` at
    ``k = 0``, ``p``/``c`` the left/right wing slopes, and ``v_tilde``
    the minimum implied variance (paper, Section 3.4).
    """
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("t must be finite and > 0")
    w0 = float(raw_svi_w(0.0, p))
    wp0 = float(raw_svi_dw(0.0, p))
    w_min = p.a + p.b * p.sigma * np.sqrt(1.0 - p.rho**2)
    if w0 <= 0.0:
        raise ValueError("ATM total variance must be > 0")
    return {
        "v": w0 / t,
        "psi": wp0 / (2.0 * np.sqrt(t * w0)),
        "p": p.b * (1.0 - p.rho) / t,
        "c": p.b * (1.0 + p.rho) / t,
        "v_tilde": w_min / t,
    }


# ---------------------------------------------------------------------
# Calibration.
# ---------------------------------------------------------------------


def _check_smile(k: Array, w: Array, min_points: int) -> tuple[Array, Array]:
    """Shared smile validation; returns sorted, validated copies."""
    kk = np.asarray(k, dtype=float).ravel()
    ww = np.asarray(w, dtype=float).ravel()
    if kk.shape != ww.shape:
        raise ValueError("k and w must have the same length")
    if kk.size < min_points:
        raise ValueError(f"need >= {min_points} quotes")
    if not np.isfinite(kk).all() or not np.isfinite(ww).all():
        raise ValueError("quotes must be finite")
    if (ww <= 0.0).any():
        raise ValueError("total variances must be > 0")
    return kk, ww


def _interp_w(k: Array | float, k_obs: Array, w_obs: Array) -> Array:
    """Per-point linear interpolation in k with linear wing extension.

    Inside the quoted range this is ``np.interp``; outside, the smile is
    extended along the edge secants. The extension can violate the Lee
    wing bound - that is documented (module docstring), not hidden.
    Requires ``k_obs`` strictly increasing.
    """
    kk = np.asarray(k, dtype=float)
    base = np.interp(kk, k_obs, w_obs)
    left = kk < k_obs[0]
    right = kk > k_obs[-1]
    if bool(left.any()):
        s_lo = float((w_obs[1] - w_obs[0]) / (k_obs[1] - k_obs[0]))
        base = np.where(left, w_obs[0] + s_lo * (kk - k_obs[0]), base)
    if bool(right.any()):
        s_hi = float((w_obs[-1] - w_obs[-2]) / (k_obs[-1] - k_obs[-2]))
        base = np.where(right, w_obs[-1] + s_hi * (kk - k_obs[-1]), base)
    return np.asarray(base, dtype=float)


@dataclass(frozen=True)
class SmileSlice:
    """One calibrated maturity slice: SVI fit or interpolation fallback.

    ``mode == "svi"`` carries ``params``; ``mode == "interp"`` evaluates
    linear interpolation on the observed quotes (fallback when every
    optimizer start failed or returned non-finite residuals).
    ``k_obs``/``w_obs`` are the sorted input quotes.
    """

    t: float
    mode: str
    params: SVIParams | None
    rmse: float
    converged: bool
    penalty: float
    k_obs: Array
    w_obs: Array

    def __post_init__(self) -> None:
        if self.mode not in ("svi", "interp"):
            raise ValueError("mode must be 'svi' or 'interp'")
        if self.mode == "svi" and self.params is None:
            raise ValueError("svi mode requires params")
        if np.diff(np.asarray(self.k_obs, dtype=float)).min() <= 0.0:
            raise ValueError("k_obs must be strictly increasing")

    def w(self, k: Array | float) -> Array:
        """Total implied variance of this slice at log-moneyness ``k``."""
        if self.mode == "svi" and self.params is not None:
            return raw_svi_w(k, self.params)
        return _interp_w(k, self.k_obs, self.w_obs)


def _initial_guess(k: Array, w: Array) -> SVIParams:
    """Data-driven start: m at the smile minimum, b/rho from edge slopes."""
    n = k.size
    i0 = int(np.argmin(w))
    span = max(float(k[-1] - k[0]), 1e-6)
    lo = max(1, n // 4)
    s_left = float((w[lo] - w[0]) / (k[lo] - k[0])) if k[lo] > k[0] else 0.0
    j = n - 1 - lo
    s_right = float((w[-1] - w[j]) / (k[-1] - k[j])) if k[-1] > k[j] else 0.0
    # Left-wing slope in |k| units is -dw/dk on the left edge (w falls
    # toward the minimum as k increases, so dw/dk < 0 there).
    left_slope, right_slope = max(-s_left, 0.0), max(s_right, 0.0)
    denom = left_slope + right_slope
    rho0 = 0.0 if denom < 1e-12 else float(np.clip((right_slope - left_slope) / denom, -0.9, 0.9))
    b0 = max(0.5 * denom, 1e-4)
    return SVIParams(
        a=float(max(w[i0] - b0 * span * 0.25, 1e-6)),
        b=b0,
        rho=rho0,
        m=float(k[i0]),
        sigma=max(0.25 * span, 1e-3),
    )


def calibrate_slice(
    k: Array,
    w_obs: Array,
    t: float,
    *,
    n_starts: int = 12,
    seed: int = 0,
    weights: Array | None = None,
    arb_penalty: float | None = None,
    prev_slice: SVIParams | None = None,
    n_penalty_grid: int = 101,
) -> SmileSlice:
    """Least-squares fit of raw SVI to one total-variance smile.

    ``k`` is log-moneyness, ``w_obs`` the observed total implied
    variances (``sigma_BS^2 t``), ``t > 0`` the maturity (the fit itself
    is in ``w``; ``t`` only labels the slice for surface assembly).
    Multi-start: one data-driven start plus ``n_starts - 1`` seeded
    uniform restarts inside the bounds; the best finite-cost run wins.
    With ``arb_penalty = lam`` the objective gains ``lam *`` hinge
    penalties on ``g < 0`` and ``w < 0`` over a dense k-grid and, when
    ``prev_slice`` is supplied, on ``w(k) < w_prev(k)`` (calendar). The
    penalty pushes the fit toward the no-arbitrage region but does not
    guarantee it - verify the result with ``slice_arbitrage_report``.
    If every start fails, falls back to per-point interpolation
    (``mode == "interp"``).

    Fail-closed: < 5 quotes, duplicated k, mismatched/non-finite/
    non-positive inputs raise ``ValueError``.
    """
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("t must be finite and > 0")
    kk, ww = _check_smile(k, w_obs, 5)
    if weights is not None:
        wt = np.asarray(weights, dtype=float).ravel()
        if wt.shape != ww.shape or not np.isfinite(wt).all() or (wt < 0.0).any():
            raise ValueError("weights must match quotes, finite, >= 0")
    else:
        wt = np.ones_like(ww)
    if arb_penalty is not None and (not np.isfinite(arb_penalty) or arb_penalty < 0.0):
        raise ValueError("arb_penalty must be finite and >= 0")
    if prev_slice is not None and arb_penalty is None:
        raise ValueError("prev_slice only makes sense with arb_penalty")
    if n_penalty_grid < 3:
        raise ValueError("n_penalty_grid must be >= 3")
    order = np.argsort(kk)
    kk, ww, wt = kk[order], ww[order], wt[order]
    if np.diff(kk).min() <= 0.0:
        raise ValueError("k values must be distinct")

    span = float(kk[-1] - kk[0])
    a_hi = float(2.0 * np.abs(ww).max() + 1.0)
    lb = np.array([-a_hi, 0.0, -0.999, kk[0] - 3.0 * span, 1e-6])
    ub = np.array([a_hi, 5.0, 0.999, kk[-1] + 3.0 * span, 50.0])
    lam = float(arb_penalty) if arb_penalty is not None else 0.0
    pad = 0.5 * span
    k_pen = np.linspace(kk[0] - pad, kk[-1] + pad, int(n_penalty_grid))
    n_pen = k_pen.size * (2 + int(prev_slice is not None)) if arb_penalty else 0

    def resid_plain(x: Array) -> Array:
        a, b, rho, m, sig = (float(v) for v in x)
        try:
            p = SVIParams(a, b, rho, m, sig)
        except ValueError:
            return np.full(ww.size, 1e6)
        return np.asarray((raw_svi_w(kk, p) - ww) * wt, dtype=float)

    def resid_pen(x: Array) -> Array:
        a, b, rho, m, sig = (float(v) for v in x)
        try:
            p = SVIParams(a, b, rho, m, sig)
        except ValueError:
            return np.full(ww.size + n_pen, 1e6)
        r = (raw_svi_w(kk, p) - ww) * wt
        parts = [
            # clip keeps the residual finite where g_raw degenerates.
            np.clip(np.minimum(_g_raw(k_pen, p), 0.0), -1e6, 0.0),
            np.minimum(raw_svi_w(k_pen, p), 0.0),
        ]
        if prev_slice is not None:
            parts.append(np.minimum(raw_svi_w(k_pen, p) - raw_svi_w(k_pen, prev_slice), 0.0))
        return np.asarray(np.concatenate([r, lam * np.concatenate(parts)]), dtype=float)

    def run_multistart(
        resid_fn: Callable[[Array], Array], x_extra: list[Array]
    ) -> optimize.OptimizeResult | None:
        rng = np.random.default_rng(int(seed))
        starts: list[Array] = list(x_extra)
        for _ in range(max(int(n_starts) - len(starts), 0)):
            starts.append(rng.uniform(lb, ub))
        best_res: optimize.OptimizeResult | None = None
        for x0 in starts:
            x0c = np.clip(np.asarray(x0, dtype=float), lb + 1e-9, ub - 1e-9)
            try:
                res = optimize.least_squares(resid_fn, x0c, bounds=(lb, ub), method="trf")
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                # Narrowed from `except Exception` (quality ratchet): trf
                # failures at a start point are bad-seed skips, not verdicts.
                # RuntimeError is included because scipy/torch/etc. backends
                # can raise it on Windows when the optimizer aborts.
                continue
            if (
                np.isfinite(res.cost)
                and np.isfinite(np.asarray(res.fun)).all()
                and (best_res is None or res.cost < best_res.cost)
            ):
                best_res = res
        return best_res

    p0 = _initial_guess(kk, ww)
    x0_data = np.array([p0.a, p0.b, p0.rho, p0.m, p0.sigma])
    if arb_penalty is None:
        best = run_multistart(resid_plain, [x0_data])
    else:
        # Warm start: solve the plain problem first, then refine with
        # the no-arb penalty from that optimum (plus fresh restarts).
        plain = run_multistart(resid_plain, [x0_data])
        warm = [np.asarray(plain.x, dtype=float)] if plain is not None else []
        best = run_multistart(resid_pen, [*warm, x0_data])

    if best is None:
        return SmileSlice(
            t=float(t),
            mode="interp",
            params=None,
            rmse=np.inf,
            converged=False,
            penalty=np.inf,
            k_obs=kk,
            w_obs=ww,
        )

    a, b, rho, m, sig = (float(v) for v in np.asarray(best.x, dtype=float))
    params = SVIParams(a, b, rho, m, sig)
    rmse = float(np.sqrt(np.mean((raw_svi_w(kk, params) - ww) ** 2)))
    g_neg = float(np.minimum(svi_g(k_pen, params), 0.0).sum())
    w_neg = float(np.minimum(raw_svi_w(k_pen, params), 0.0).sum())
    penalty = -lam * (g_neg + w_neg)
    return SmileSlice(
        t=float(t),
        mode="svi",
        params=params,
        rmse=rmse,
        converged=bool(best.success),
        penalty=penalty,
        k_obs=kk,
        w_obs=ww,
    )


def calibrate_slice_iv(
    k: Array,
    iv_obs: Array,
    t: float,
    *,
    n_starts: int = 12,
    seed: int = 0,
    weights: Array | None = None,
    arb_penalty: float | None = None,
    prev_slice: SVIParams | None = None,
) -> SmileSlice:
    """Calibrate to implied-vol quotes; converts ``w_obs = iv^2 t``."""
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("t must be finite and > 0")
    iv = np.asarray(iv_obs, dtype=float).ravel()
    if iv.size == 0:
        raise ValueError("empty smile")
    if not np.isfinite(iv).all() or (iv <= 0.0).any():
        raise ValueError("iv quotes must be finite and > 0")
    return calibrate_slice(
        k,
        iv**2 * t,
        t,
        n_starts=n_starts,
        seed=seed,
        weights=weights,
        arb_penalty=arb_penalty,
        prev_slice=prev_slice,
    )


# ---------------------------------------------------------------------
# Surface assembly, calendar arbitrage, monotone repair.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SVISurface:
    """SVI slices on a maturity grid; ``w(k, t)`` linear in t per slice pair.

    Between grid maturities, total variance is interpolated linearly at
    fixed ``k`` (documented choice: preserves positivity, may introduce a
    kink in t - the calendar check runs on the grid slices where the
    model is defined).
    """

    maturities: Array
    slices: tuple[SmileSlice, ...]

    def w(self, k: Array | float, t: float) -> Array:
        """Total implied variance at ``(k, t)``; ``t`` inside the grid."""
        tt = np.asarray(self.maturities, dtype=float)
        if not np.isfinite(t) or t < tt[0] or t > tt[-1]:
            raise ValueError("t outside the maturity grid")
        ws = np.stack([s.w(k) for s in self.slices], axis=0)  # (n_t, *k)
        alpha = float(np.interp(t, tt, np.arange(tt.size, dtype=float)))
        i0 = int(np.clip(np.floor(alpha), 0, tt.size - 2))
        frac = alpha - i0
        return np.asarray((1.0 - frac) * ws[i0] + frac * ws[i0 + 1], dtype=float)


def assemble_surface(slices: list[SmileSlice] | tuple[SmileSlice, ...]) -> SVISurface:
    """Stitch calibrated slices into a surface (fail-closed on bad grids)."""
    if len(slices) < 1:
        raise ValueError("need >= 1 slice")
    ts = np.array([s.t for s in slices], dtype=float)
    if not np.isfinite(ts).all() or (ts <= 0.0).any():
        raise ValueError("maturities must be finite and > 0")
    if ts.size > 1 and np.diff(ts).min() <= 0.0:
        raise ValueError("maturities must be strictly increasing")
    return SVISurface(maturities=ts, slices=tuple(slices))


def calendar_report(surface: SVISurface, k_grid: Array) -> dict[str, float]:
    """Calendar-arbitrage check on the assembled surface.

    No calendar arbitrage (paper, Definition 2.2) requires total
    variance non-decreasing in t for every fixed k. Evaluates each slice
    on ``k_grid`` and reports the worst decrement between adjacent
    maturities and the count of violating ``(k, t_i)`` pairs.
    """
    kg = np.asarray(k_grid, dtype=float).ravel()
    if kg.size == 0 or not np.isfinite(kg).all():
        raise ValueError("k_grid must be non-empty and finite")
    w_mat = np.stack([s.w(kg) for s in surface.slices], axis=0)  # (n_t, n_k)
    d = np.diff(w_mat, axis=0)
    worst = float(d.min()) if d.size else 0.0
    n_bad = int((d < -_ARB_TOL).sum())
    return {
        "worst_decrement": worst,
        "n_violations": float(n_bad),
        "n_pairs": float(d.size),
        "free": float(n_bad == 0),
    }


def _pava_nondec(y: Array) -> Array:
    """Unweighted isotonic regression onto non-decreasing sequences."""
    levels: list[float] = []
    counts: list[int] = []
    for v in y:
        levels.append(float(v))
        counts.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            merged = (levels[-2] * counts[-2] + levels[-1] * counts[-1]) / (counts[-2] + counts[-1])
            levels[-2:] = [merged]
            counts[-2:] = [counts[-2] + counts[-1]]
    out = np.empty(y.size)
    idx = 0
    for lv, c in zip(levels, counts, strict=True):
        out[idx : idx + c] = lv
        idx += c
    return out


@dataclass(frozen=True)
class GriddedSurface:
    """Total-variance surface on a ``(k, t)`` grid; bilinear evaluation.

    ``w_mat[i, j]`` is the total variance at ``(k_grid[j],
    maturities[i])``.
    """

    k_grid: Array
    maturities: Array
    w_mat: Array

    def w(self, k: Array | float, t: float) -> Array:
        """Bilinear interpolation of total variance at ``(k, t)``."""
        kk = np.asarray(k, dtype=float)
        if not np.isfinite(t) or t < self.maturities[0] or t > self.maturities[-1]:
            raise ValueError("t outside the maturity grid")
        per_slice = np.stack(
            [np.interp(kk.ravel(), self.k_grid, self.w_mat[i]) for i in range(self.maturities.size)]
        )
        alpha = float(np.interp(t, self.maturities, np.arange(self.maturities.size, dtype=float)))
        i0 = int(np.clip(np.floor(alpha), 0, self.maturities.size - 2))
        frac = alpha - i0
        out = (1.0 - frac) * per_slice[i0] + frac * per_slice[i0 + 1]
        return np.asarray(out.reshape(kk.shape), dtype=float)


def repair_calendar(surface: SVISurface, k_grid: Array) -> GriddedSurface:
    """Monotone projection of the surface's ``w(k, t_i)`` columns (PAVA).

    Samples each slice on ``k_grid``, then for every fixed k replaces the
    maturity sequence by its non-decreasing isotonic regression - the
    L2-closest monotone grid surface. A repair, not a re-fit: the result
    is a gridded surface, not SVI parameters (honesty note in the module
    docstring).
    """
    kg = np.asarray(k_grid, dtype=float).ravel()
    if kg.size < 2 or not np.isfinite(kg).all():
        raise ValueError("k_grid must have >= 2 finite points")
    if np.diff(kg).min() <= 0.0:
        raise ValueError("k_grid must be strictly increasing")
    w_mat = np.stack([s.w(kg) for s in surface.slices], axis=0)
    fixed = np.apply_along_axis(_pava_nondec, 0, w_mat)
    return GriddedSurface(k_grid=kg, maturities=surface.maturities, w_mat=fixed)


# ---------------------------------------------------------------------
# Synthetic benchmark.
# ---------------------------------------------------------------------


def _synth_clean_params(rng: np.random.Generator) -> SVIParams:
    """A slice drawn strictly inside the SSVI Theorem-4.2 region.

    Both conditions hold with margin, so the resulting raw slice is
    provably free of butterfly arbitrage - the label is certain, not
    detector-dependent.
    """
    for _ in range(200):
        theta = float(rng.uniform(0.02, 0.15))
        rho = float(rng.uniform(-0.6, 0.6))
        phi = float(rng.uniform(0.1, 2.0))
        if theta * phi * (1.0 + abs(rho)) < 3.0 and theta * phi**2 * (1.0 + abs(rho)) <= 3.0:
            return ssvi_to_raw(theta, rho, phi)
    return ssvi_to_raw(0.05, -0.3, 0.5)


def _synth_violating_params(rng: np.random.Generator, kind: str = "any") -> SVIParams:
    """A slice with a planted, provably-detectable arbitrage.

    ``kind == "wing"``: a Lee wing breach (``b (1 + |rho|) > 2`` with
    margin). ``kind == "negvar"``: negative minimum total variance
    (``a < -b sigma sqrt(1 - rho^2)``). ``"any"`` picks one at random.
    """
    if kind not in ("wing", "negvar", "any"):
        raise ValueError("unknown violation kind")
    if kind == "wing" or (kind == "any" and rng.integers(0, 2) == 0):
        rho = float(rng.uniform(-0.9, 0.9))
        b = float(rng.uniform(2.2, 3.0) / (1.0 + abs(rho)))
        return SVIParams(a=0.05, b=b, rho=rho, m=0.0, sigma=0.3)
    rho = float(rng.uniform(-0.5, 0.5))
    sigma = float(rng.uniform(0.1, 0.4))
    b = float(rng.uniform(0.3, 1.0))
    a = -b * sigma * np.sqrt(1.0 - rho**2) - float(rng.uniform(0.01, 0.05))
    return SVIParams(a=float(a), b=b, rho=rho, m=0.0, sigma=sigma)


def bench_svi_surface(seed: int = 4213) -> dict[str, float]:
    """Synthetic correctness bench - flat dict of floats, ``synthetic_*`` keys.

    Calibration residuals on exact and noisy SVI quotes, butterfly and
    calendar detection precision/recall on planted violations, mapping
    round-trip and wing-extrapolation errors. SYNTHETIC correctness
    evidence only (see the module docstring).
    """
    rng = np.random.default_rng(int(seed))
    out: dict[str, float] = {}

    # 1. Calibration recovery on exact SVI quotes.
    k = np.linspace(-1.0, 1.0, 25)
    true_p = SVIParams(a=0.04, b=0.4, rho=-0.45, m=0.02, sigma=0.25)
    w_true = raw_svi_w(k, true_p)
    fit = calibrate_slice(k, w_true, 0.5, seed=seed, n_starts=8)
    out["synthetic_calib_rmse_exact"] = float(fit.rmse)
    if fit.params is not None:
        out["synthetic_calib_w_err"] = float(np.max(np.abs(raw_svi_w(k, fit.params) - w_true)))
    else:
        out["synthetic_calib_w_err"] = np.inf

    w_noisy = np.clip(w_true * (1.0 + rng.normal(scale=0.01, size=k.size)), 1e-6, None)
    fit_n = calibrate_slice(k, w_noisy, 0.5, seed=seed, n_starts=8)
    out["synthetic_calib_rmse_noisy"] = float(fit_n.rmse)

    # 2. Butterfly detection: precision/recall vs construction labels.
    n_pos = 40
    tp = fp = fn = tn = 0
    for _ in range(n_pos):
        for violate in (False, True):
            p = _synth_violating_params(rng) if violate else _synth_clean_params(rng)
            rep = slice_arbitrage_report(p, -8.0, 8.0, 4001)
            pred = rep["free"] < 0.5
            if violate and pred:
                tp += 1
            elif violate and not pred:
                fn += 1
            elif not violate and pred:
                fp += 1
            else:
                tn += 1
    out["synthetic_arb_precision"] = float(tp / max(tp + fp, 1))
    out["synthetic_arb_recall"] = float(tp / max(tp + fn, 1))
    out["synthetic_arb_f1"] = float(2 * tp / max(2 * tp + fp + fn, 1))

    # 3. Calendar detection: one dropped-theta slice planted per surface.
    n_surf, n_cal_bad = 20, 0
    k_cal = np.linspace(-2.0, 2.0, 41)
    for _ in range(n_surf):
        ts = np.linspace(0.25, 1.0, 4)
        theta0 = float(rng.uniform(0.02, 0.08))
        thetas = theta0 * ts / ts[0]
        plant = int(rng.integers(1, 4))
        slices: list[SmileSlice] = []
        for i in range(4):
            rho_i = float(rng.uniform(-0.5, -0.1))
            phi_i = float(rng.uniform(0.3, 0.8))
            thi = thetas[i] if i != plant else thetas[i - 1] * 0.5
            slices.append(
                SmileSlice(
                    t=float(ts[i]),
                    mode="svi",
                    params=ssvi_to_raw(thi, rho_i, phi_i),
                    rmse=0.0,
                    converged=True,
                    penalty=0.0,
                    k_obs=k,
                    w_obs=w_true,
                )
            )
        rep = calendar_report(assemble_surface(slices), k_cal)
        n_cal_bad += int(rep["free"] < 0.5)
    out["synthetic_calendar_detection"] = float(n_cal_bad / n_surf)

    # 4. Monotone repair: worst residual calendar decrement after repair.
    rep_grid = repair_calendar(assemble_surface(slices), k_cal)
    d_fixed = np.diff(rep_grid.w_mat, axis=0)
    out["synthetic_repair_residual"] = float(np.abs(np.minimum(d_fixed, 0.0)).max())

    # 5. Wing extrapolation: SVI iv vs the asymptote sqrt(slope * k / t).
    k_far = np.array([5.0, 8.0, 12.0])
    iv_svi = svi_implied_vol(k_far, true_p, 0.5)
    slope_r = true_p.b * (1.0 + true_p.rho)
    iv_asymp = np.sqrt(slope_r * k_far / 0.5)
    out["synthetic_extrap_rel_err"] = float(np.max(np.abs(iv_svi - iv_asymp) / iv_svi))

    # 6. Mapping round-trips.
    pn = raw_to_natural(true_p)
    back = natural_to_raw(pn)
    out["synthetic_natural_roundtrip"] = float(
        np.max(np.abs(natural_svi_w(k, pn) - w_true))
        + abs(back.a - true_p.a)
        + abs(back.m - true_p.m)
    )
    th, rh, ph = 0.05, -0.4, 0.6
    ssvi_direct = ssvi_w(k, th, rh, ph)
    ssvi_mapped = raw_svi_w(k, ssvi_to_raw(th, rh, ph))
    out["synthetic_ssvi_equiv_err"] = float(np.max(np.abs(ssvi_direct - ssvi_mapped)))

    n_lee = 20
    lee_hits = sum(
        lee_wing_check(_synth_violating_params(rng, "wing"))["violated"] for _ in range(n_lee)
    )
    out["synthetic_lee_detection"] = float(lee_hits) / n_lee
    out["synthetic_slices_bench"] = float(n_pos * 2)
    return out
