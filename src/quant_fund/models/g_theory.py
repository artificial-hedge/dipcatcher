"""Generalizability theory — single-facet person x rater G-study
and D-study.

Brennan (2001), Cronbach, Gleser, Nanda & Rajaratnam (1972): for a
crossed person x rater design (n persons, k raters) the two-way
random-effects ANOVA yields variance components

    sigma_p^2 = (MS_p - MS_e) / k       person (universe score)
    sigma_r^2 = (MS_r - MS_e) / n       rater
    sigma_e^2 = MS_e                    residual (p x r + error)

and the G-coefficients
    E rho^2 = sigma_p^2 / (sigma_p^2 + sigma_delta^2)   relative
    Phi     = sigma_p^2 / (sigma_p^2 + sigma_Delta^2)   absolute
with sigma_delta^2 = sigma_e^2/k (relative error) and
sigma_Delta^2 = (sigma_r^2 + sigma_e^2)/k (absolute error). A
D-study recomputes the coefficients for a proposed k' raters.

Honesty: components are clipped at zero (standard practice —
negative ANOVA components are set to zero and reported). The
bench uses planted person variance + small rater effects (G high)
and pure noise (G low). Fail-closed on degenerate mean squares.

References: Cronbach, Gleser, Nanda, Rajaratnam (1972) "The
dependability of behavioral measurements"; Brennan (2001)
"Generalizability Theory"; Shavelson & Webb (1991).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check(x: FloatArray) -> FloatArray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] < 5 or a.shape[1] < 2:
        raise ValueError("bad design")
    if not np.isfinite(a).all():
        raise ValueError("non-finite input")
    return a


def g_study(x: FloatArray) -> dict[str, float]:
    """Single-facet G-study variance components + coefficients."""
    a = _check(x)
    n, k = a.shape
    grand = a.mean()
    rp = a.mean(axis=1)  # person means
    rr = a.mean(axis=0)  # rater means
    ss_p = k * float(((rp - grand) ** 2).sum())
    ss_r = n * float(((rr - grand) ** 2).sum())
    ss_t = float(((a - grand) ** 2).sum())
    ss_e = ss_t - ss_p - ss_r
    ms_p = ss_p / (n - 1)
    ms_r = ss_r / (k - 1)
    ms_e = ss_e / max((n - 1) * (k - 1), 1)
    s_p = max(0.0, (ms_p - ms_e) / k)
    s_r = max(0.0, (ms_r - ms_e) / n)
    s_e = ms_e
    rel_err = s_e / k
    abs_err = (s_r + s_e) / k
    denom_g = s_p + rel_err
    denom_phi = s_p + abs_err
    g_coeff = s_p / denom_g if denom_g > 0 else 0.0
    phi = s_p / denom_phi if denom_phi > 0 else 0.0
    return {
        "sigma_person": float(s_p),
        "sigma_rater": float(s_r),
        "sigma_residual": float(s_e),
        "g_coefficient": float(g_coeff),
        "phi_coefficient": float(phi),
        "relative_error": float(rel_err),
        "absolute_error": float(abs_err),
    }


def d_study(x: FloatArray, n_raters: int = 6) -> dict[str, float]:
    """D-study: recompute G/Phi for a proposed ``n_raters``."""
    if n_raters < 1:
        raise ValueError("need >=1 rater")
    a = _check(x)
    comp = g_study(a)
    s_p = comp["sigma_person"]
    s_r = comp["sigma_rater"]
    s_e = comp["sigma_residual"]
    rel_err = s_e / n_raters
    abs_err = (s_r + s_e) / n_raters
    denom_g = s_p + rel_err
    denom_phi = s_p + abs_err
    return {
        "g_coefficient": float(s_p / denom_g) if denom_g > 0 else 0.0,
        "phi_coefficient": float(s_p / denom_phi) if denom_phi > 0 else 0.0,
        "n_raters": float(n_raters),
    }


def bench_g_theory(seed: int = 20261231 + 441) -> dict[str, float]:
    """SYNTHETIC check — person variance dominates, D-study improves G."""
    rng = np.random.default_rng(seed)
    n, k = 80, 3
    person = rng.standard_normal(n) * 2.0
    rater = np.array([0.0, 0.3, -0.2])
    x = person[:, None] + rater[None, :] + 0.5 * rng.standard_normal((n, k))
    out = g_study(x)
    noise = g_study(rng.standard_normal((n, k)))
    d = d_study(x, n_raters=10)
    if (
        out["g_coefficient"] < 0.85
        or noise["g_coefficient"] > 0.4
        or d["g_coefficient"] <= out["g_coefficient"]
    ):
        raise ValueError(
            f"g_theory off: g={out['g_coefficient']:.3f} noise={noise['g_coefficient']:.3f} "
            f"d10={d['g_coefficient']:.3f}"
        )
    return {
        "synthetic_g": out["g_coefficient"],
        "synthetic_phi": out["phi_coefficient"],
        "synthetic_g_noise": noise["g_coefficient"],
        "synthetic_d_g_10": d["g_coefficient"],
        "score": 1.0,
    }
