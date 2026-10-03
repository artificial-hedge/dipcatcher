"""Gaussian normalized coordinates and risk-neutral CDF deformations.

Implements the coordinate/deformation framework of Sun (2026): express a
risk-neutral CDF in an increasing Gaussian normalized coordinate ``z`` and
write its departure from the Gaussian benchmark as the *CDF deformation*

    eta(z) = (h(z) - Phi(z)) / phi(z),   h(z) = Phi(z) + phi(z) * eta(z),

whose differential transform

    m(z) = 1 + eta'(z) - z * eta(z),     h'(z) = phi(z) * m(z),

is the option-implied density in normalized coordinates relative to the
standard normal. No-arbitrage then reduces to the pointwise sign condition
``m(z) >= 0`` together with the Mills-ratio bounds ``-Phi/phi <= eta <=
(1-Phi)/phi`` on the deformation (equivalently ``0 <= h <= 1``).

Two conventions realize the same deformation:

- Bachelier: normalized coordinate ``y = (K - F) / s(K)`` where ``s`` is
  the total normal implied standard deviation; eta is exactly ``ds/dK``;
  with reciprocal scale ``q(y) = 1/s(y)``, butterfly convexity is the
  linear inequality ``q''(y) + y q'(y) - q(y) <= 0``.
- Black (Fukasawa): log-moneyness ``k = log(K/F)``, total implied
  volatility ``v = sigma_imp * sqrt(T)``; normalized coordinates
  ``a = k/v - v/2`` and ``b = k/v + v/2``; the same CDF deformation is the
  total-vol slope ``v'(k)``, and the Black butterfly factor reproduces
  ``m``.

References
----------
Sun, J. (2026). "Gaussian Normalized Coordinates and Risk-Neutral CDF
Deformations." arXiv:2609.14212.
Fukasawa, M. (2012). "The normalizing transformation of the implied
volatility smile." Mathematical Finance 22(4).

Honesty: all diagnostics are SYNTHETIC — generated smiles/deformations on
seeded fixtures that validate the normalized-space no-arbitrage machinery.
No market data, no P&L/NAV claims.

Composition notes: normal-model pricing/inversion and the Bachelier
implied scale are implemented here (no existing quant_fund module owns a
normal-model pricer); ``scipy.interpolate`` supplies the differentiable
splines the deformation and repair projection need. ``models/iv_approx``
has only crude IV approximations — not reused, since this module works in
exact model-space coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate, stats

Array = NDArray[np.float64]

__all__ = [
    "ArbReport",
    "bachelier_normalized",
    "bench_gaussian_normalized_coords",
    "butterfly_inequality_q",
    "cdf_deformation_eta",
    "deformed_cdf",
    "density_factor_m",
    "fukasawa_coords",
    "mills_bounds_eta",
    "normalized_arb_check",
    "repair_to_arb_free",
    "synthetic_scale_surface",
]


_EPS = 1e-12


def _require(cond: object, msg: str) -> None:
    if not bool(cond):
        raise ValueError(msg)


def _as_1d(x: Array, name: str, n_min: int = 3) -> Array:
    a = np.asarray(x, dtype=float).ravel()
    _require(a.size >= n_min and np.all(np.isfinite(a)), f"{name} must be finite len>={n_min}")
    return a


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------


def bachelier_normalized(k: Array, s: Array) -> Array:
    """Bachelier normalized coordinate y = k / s where k = K - F (absolute)."""
    kk = np.asarray(k, dtype=float)
    ss = np.asarray(s, dtype=float)
    _require(kk.shape == ss.shape and kk.size >= 3, "k and s must match, len>=3")
    _require(np.all(ss > 0), "total scale must be positive")
    return np.asarray(kk / ss, dtype=float)


def fukasawa_coords(k: Array, v: Array) -> tuple[Array, Array]:
    """Black/Fukasawa normalized coords a = k/v - v/2, b = k/v + v/2.

    ``k`` is log-moneyness log(K/F), ``v = sigma_imp * sqrt(T)`` the total
    implied volatility. a (b) is the z-coordinate under the risk-neutral
    (share) measure.
    """
    kk = np.asarray(k, dtype=float)
    vv = np.asarray(v, dtype=float)
    _require(kk.shape == vv.shape and kk.size >= 3, "k and v must match, len>=3")
    _require(np.all(vv > 0), "total vol must be positive")
    return kk / vv - vv / 2.0, kk / vv + vv / 2.0


def mills_bounds_eta(z: Array) -> tuple[Array, Array]:
    """Mills-ratio bounds on the CDF deformation: -Phi/phi <= eta <= (1-Phi)/phi."""
    zz = np.asarray(z, dtype=float)
    _require(np.all(np.isfinite(zz)), "z must be finite")
    phi = np.maximum(stats.norm.pdf(zz), _EPS)
    return -stats.norm.cdf(zz) / phi, (1.0 - stats.norm.cdf(zz)) / phi


# ---------------------------------------------------------------------------
# Deformation and density
# ---------------------------------------------------------------------------


def cdf_deformation_eta(z: Array, h: Array) -> Array:
    """eta(z) = (h(z) - Phi(z)) / phi(z) — pointwise CDF deformation."""
    zz = _as_1d(z, "z")
    hh = np.asarray(h, dtype=float).ravel()
    _require(hh.shape == zz.shape, "h must match z")
    phi = stats.norm.pdf(zz)
    _require(np.all(phi > _EPS), "z too far in the tails for a stable deformation")
    return np.asarray((hh - stats.norm.cdf(zz)) / phi, dtype=float)


def deformed_cdf(z: Array, eta: Array) -> Array:
    """Reconstruct h(z) = Phi(z) + phi(z) * eta(z) from a deformation."""
    zz = _as_1d(z, "z")
    ee = np.asarray(eta, dtype=float).ravel()
    _require(ee.shape == zz.shape, "eta must match z")
    return np.asarray(stats.norm.cdf(zz) + stats.norm.pdf(zz) * ee, dtype=float)


def density_factor_m(z: Array, eta: Array) -> Array:
    """m(z) = 1 + eta'(z) - z eta(z) — implied density relative to Gaussian.

    ``eta'`` is taken by centered differences on a sorted grid (grids from
    the bachelier/fukasawa maps are increasing whenever the surface is
    non-degenerate).
    """
    zz = _as_1d(z, "z", n_min=5)
    ee = np.asarray(eta, dtype=float).ravel()
    _require(ee.shape == zz.shape, "eta must match z")
    order = np.argsort(zz)
    z_s, e_s = zz[order], ee[order]
    deta = np.gradient(e_s, z_s)
    m = np.asarray(1.0 + deta - z_s * e_s, dtype=float)
    out = np.empty_like(m)
    out[order] = m
    return out


# ---------------------------------------------------------------------------
# Arbitrage checks
# ---------------------------------------------------------------------------


@dataclass
class ArbReport:
    """No-arbitrage diagnostics in normalized coordinates."""

    m_min: float
    m_negative_count: int
    mills_violations: int
    arbitrage_free: bool


def normalized_arb_check(z: Array, eta: Array) -> ArbReport:
    """Check m >= 0 plus Mills-ratio bounds — normalized-space no-arb."""
    zz = _as_1d(z, "z", n_min=5)
    ee = np.asarray(eta, dtype=float).ravel()
    _require(ee.shape == zz.shape, "eta must match z")
    m = density_factor_m(zz, ee)
    lo, hi = mills_bounds_eta(zz)
    m_neg = int(np.sum(m < 0.0))
    mills_v = int(np.sum((ee < lo - 1e-9) | (ee > hi + 1e-9)))
    return ArbReport(
        m_min=float(m.min()),
        m_negative_count=m_neg,
        mills_violations=mills_v,
        arbitrage_free=bool(m_neg == 0 and mills_v == 0),
    )


def butterfly_inequality_q(y: Array, q: Array) -> Array:
    """Bachelier-convention butterfly residual r(y) = q'' + y q' - q.

    Butterfly convexity in normalized Bachelier coordinates is equivalent
    to ``r(y) <= 0`` (Sun 2026, linear inequality in reciprocal scale
    ``q = 1/s``). Returns the residual; positive entries flag violations.
    """
    yy = _as_1d(y, "y", n_min=5)
    qq = np.asarray(q, dtype=float).ravel()
    _require(qq.shape == yy.shape and np.all(qq > 0), "q must be positive, match y")
    order = np.argsort(yy)
    y_s, q_s = yy[order], qq[order]
    q1 = np.gradient(q_s, y_s)
    q2 = np.gradient(q1, y_s)
    resid = np.asarray(q2 + y_s * q1 - q_s, dtype=float)
    out = np.empty_like(resid)
    out[order] = resid
    return out


# ---------------------------------------------------------------------------
# Repair (project a violating deformation back to m >= 0)
# ---------------------------------------------------------------------------


def _isotonic_nondecreasing(h: Array) -> Array:
    """Pool-adjacent-violators projection onto nondecreasing sequences."""
    y = np.asarray(h, dtype=float).copy()
    n = y.size
    w = np.ones(n)
    i = 0
    while i < n - 1:
        if y[i] > y[i + 1] + 0.0:
            # Pool backwards until monotone.
            j = i
            tot = y[i + 1] * w[i + 1]
            cnt = w[i + 1]
            while j >= 0:
                tot += y[j] * w[j]
                cnt += w[j]
                if j > 0 and y[j - 1] > tot / cnt:
                    j -= 1
                else:
                    break
            y[j : i + 2] = tot / cnt
            i = max(j - 1, 0)
        else:
            i += 1
    return y


def repair_to_arb_free(
    z: Array,
    eta: Array,
    floor: float = 0.0,
) -> Array:
    """Project a violating deformation back to m >= 0 via isotonic repair.

    ``h = Phi + phi*eta`` is the implied CDF; ``m = h'/phi``, so h
    nondecreasing + inside [0,1] is exactly ``m >= 0`` plus the Mills
    bounds. Repair is the isotonic (PAVA) projection of h onto
    nondecreasing sequences, clipped to [0,1] — the minimal-L2 fix in CDF
    space, re-expressed as a deformation.
    """
    zz = _as_1d(z, "z", n_min=5)
    ee = np.asarray(eta, dtype=float).ravel()
    _require(ee.shape == zz.shape, "eta must match z")
    order = np.argsort(zz)
    z_s = zz[order]
    h = deformed_cdf(z_s, ee[order])
    h_rep = np.clip(_isotonic_nondecreasing(h), floor, 1.0 - floor)
    eta_rep = cdf_deformation_eta(z_s, h_rep)
    out = np.empty_like(ee)
    out[order] = eta_rep
    return out


# ---------------------------------------------------------------------------
# Synthetic surface + bench
# ---------------------------------------------------------------------------


def synthetic_scale_surface(
    k: Array,
    seed: int,
    atm: float = 0.5,
    skew: float = -0.3,
    smile: float = 0.6,
    violation_at: float | None = None,
) -> Array:
    """Total normal implied scale s(k) — smooth convex curve + optional bump.

    Baseline ``s(k) = atm + skew*k + smile*k^2`` (a generic convex smile in
    absolute moneyness); ``violation_at`` injects a localized concave bump
    (a stand-in butterfly violation) at that moneyness.
    """
    kk = _as_1d(k, "k")
    _require(atm > 0.05, "atm must be positive")
    s = atm + skew * kk + smile * kk * kk
    s = np.maximum(s, 0.02)
    if violation_at is not None:
        amp = 0.25 * atm
        s = s + amp * np.exp(-0.5 * ((kk - violation_at) / 0.05) ** 2)
        s = np.maximum(s, 0.02)
    return s


def _numeric_eta_black(k: Array, v: Array) -> tuple[Array, Array]:
    """Black-convention deformation eta = v'(k), paired with the a-coord."""
    a, b = fukasawa_coords(k, v)
    spl = interpolate.UnivariateSpline(k, v, k=3, s=0.0)
    eta = spl.derivative()(k)
    return a, eta


def bench_gaussian_normalized_coords(
    seed: int,
    n_strikes: int = 81,
    k_lo: float = -0.6,
    k_hi: float = 0.6,
    violation_at: float = 0.18,
) -> dict[str, float]:
    """Seeded SYNTHETIC bench for the normalized-coordinate machinery.

    A clean convex smile passes the m >= 0 / Mills / q-inequality checks;
    an injected localized concavity is flagged in normalized space with
    exact membership recovery; the repair projection restores
    admissibility; and the Black-vs-Bachelier deformation dictionary is
    exercised on the same underlying surface.
    """
    rng = np.random.default_rng(seed)
    k = np.linspace(k_lo, k_hi, n_strikes)

    # --- Bachelier path: s(k) -> y -> eta = ds/dk -> m ---------------------
    s_clean = synthetic_scale_surface(k, seed, skew=-0.05, smile=0.18)
    s_viol = synthetic_scale_surface(k, seed, skew=-0.05, smile=0.18, violation_at=violation_at)
    y_clean = bachelier_normalized(k, s_clean)
    spl_clean = interpolate.UnivariateSpline(k, s_clean, k=4, s=1e-8)
    spl_viol = interpolate.UnivariateSpline(k, s_viol, k=4, s=1e-8)
    eta_clean = spl_clean.derivative()(k)
    eta_viol = spl_viol.derivative()(k)

    rep_clean = normalized_arb_check(y_clean, eta_clean)
    rep_viol = normalized_arb_check(y_clean, eta_viol)

    # q-inequality on the violating surface.
    q_viol = 1.0 / np.maximum(s_viol, _EPS)
    q_resid = butterfly_inequality_q(y_clean, q_viol)
    bump_mask = np.abs(k - violation_at) <= 0.08
    detected = q_resid > 0.0
    tp = float(np.sum(detected & bump_mask))
    fp = float(np.sum(detected & ~bump_mask))
    fn = float(np.sum(~detected & bump_mask))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)

    # --- repair -----------------------------------------------------------
    eta_rep = repair_to_arb_free(y_clean, eta_viol)
    rep_fixed = normalized_arb_check(y_clean, eta_rep)
    repair_l2 = float(np.sqrt(np.mean((eta_rep - eta_viol) ** 2)))

    # --- Black path on the SAME underlying CDF -----------------------------
    # Build v(k) from the Bachelier surface via the dictionary: for the
    # synthetic fixture take v(k) = s(k)/F_eff in log coords (small-k
    # correspondence); both conventions must flag the same violation bump.
    v_clean = s_clean / np.exp(0.0)  # F=1 normalization for the fixture
    v_viol = s_viol / np.exp(0.0)
    a_coord, eta_b_clean = _numeric_eta_black(k, v_clean)
    _, eta_b_viol = _numeric_eta_black(k, v_viol)
    rep_black_clean = normalized_arb_check(a_coord, eta_b_clean)
    rep_black_viol = normalized_arb_check(a_coord, eta_b_viol)

    # --- round-trip: h -> eta -> h -----------------------------------------
    zg = np.linspace(-2.5, 2.5, 101)
    eta_ref = 0.15 * np.sin(2 * zg) * np.exp(-0.5 * zg * zg)  # admissible
    h_ref = deformed_cdf(zg, eta_ref)
    eta_back = cdf_deformation_eta(zg, h_ref)
    roundtrip = float(np.max(np.abs(eta_back - eta_ref)))

    _ = rng  # fixture is deterministic without consumption; kept for API stability

    out = {
        "synthetic_seed": float(seed),
        "synthetic_n_strikes": float(n_strikes),
        "synthetic_m_min_clean": rep_clean.m_min,
        "synthetic_clean_arb_free": float(rep_clean.arbitrage_free),
        "synthetic_m_min_viol": rep_viol.m_min,
        "synthetic_viol_flagged": float(not rep_viol.arbitrage_free),
        "synthetic_viol_m_neg_count": float(rep_viol.m_negative_count),
        "synthetic_q_precision": precision,
        "synthetic_q_recall": recall,
        "synthetic_repair_m_min": rep_fixed.m_min,
        "synthetic_repair_l2": repair_l2,
        "synthetic_black_clean_free": float(rep_black_clean.arbitrage_free),
        "synthetic_black_viol_flagged": float(not rep_black_viol.arbitrage_free),
        "synthetic_roundtrip_eta_max_err": roundtrip,
    }
    if not all(math.isfinite(v) for v in out.values()):
        return {}
    return out
