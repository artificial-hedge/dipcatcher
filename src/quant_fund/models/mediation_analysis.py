"""Causal mediation analysis — ACME/ADE with quasi-Bayesian intervals.

Fits a mediator model (M ~ T + X) and an outcome model
(Y ~ T + M + X), then simulates counterfactual mediations to estimate
the average causal mediation effect (ACME), average direct effect
(ADE), total effect, and proportion mediated, with percentile
intervals from coefficient draws (Imai's quasi-Bayesian Monte Carlo).

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure effect decomposition on generated
mediation panels — never market evidence.

References:
- Imai, Keele, Tingley (2010). A general approach to causal
  mediation analysis. *Psychological Methods* 15, 309-334.
- Baron, Kenny (1986). The moderator-mediator variable distinction.
  *JPSP* 51 (product-of-coefficients baseline).
- Tingley, Yamamoto, Hirose, Keele, Imai (2014). mediation: R
  package for causal mediation analysis. *JSS* 59.
- VanderWeele (2015). *Explanation in Causal Inference*, ch. 2.

Composition: pure numpy — OLS mediator/outcome models, multivariate-
normal coefficient draws for quasi-Bayesian intervals; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as1(v: FloatArray, name: str) -> FloatArray:
    a = np.asarray(v, dtype=np.float64).ravel()
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite 1-D vector required")
    return a


def _ols(y: FloatArray, X: FloatArray) -> tuple[FloatArray, float, FloatArray]:
    """OLS coefs + residual sd + (X'X)^-1."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    k = X.shape[1]
    s2 = float(r @ r) / max(y.size - k, 1)
    xtxi = np.linalg.pinv(X.T @ X)
    return b, math.sqrt(max(s2, 0.0)), xtxi


def mediation_analysis(
    treat: FloatArray,
    mediator: FloatArray,
    outcome: FloatArray,
    covariates: FloatArray | None = None,
    n_sims: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Estimate ACME/ADE for a binary or continuous treatment.

    Linear models, no T×M interaction: ACME = β_m·β_y(M), ADE = β_y(T),
    computed per coefficient draw from the joint asymptotic normal
    (quasi-Bayesian). Intervals are 2.5/97.5 percentiles of the draws."""
    ta = _as1(treat, "treat")
    ma = _as1(mediator, "mediator")
    ya = _as1(outcome, "outcome")
    n = ya.size
    if ta.size != n or ma.size != n:
        raise ValueError("treat/mediator/outcome length mismatch")
    if np.std(ta) < 1e-12:
        raise ValueError("treat has no variance")
    if np.std(ma) < 1e-12:
        raise ValueError("mediator has no variance")
    X = np.ones((n, 1))
    if covariates is not None:
        C = np.asarray(covariates, dtype=np.float64)
        if C.ndim != 2 or C.shape[0] != n or not np.all(np.isfinite(C)):
            raise ValueError("covariates: finite (n,k) matrix required")
        X = np.column_stack([X, C])
    if n < X.shape[1] + 8:
        raise ValueError("n too small for covariate count")

    # mediator model: M ~ T + X  (a = coef on T)
    Xm = np.column_stack([X, ta])
    bm, s_m, xm_i = _ols(ma, Xm)
    a_coef = float(bm[-1])

    # outcome model: Y ~ T + M + X  (b = coef on M, c' = coef on T)
    Xy = np.column_stack([X, ma, ta])
    by, s_y, xy_i = _ols(ya, Xy)
    b_coef = float(by[-2])
    c_prime = float(by[-1])

    rng = np.random.default_rng(seed)
    # joint coefficient draws (block-diagonal across the two models)
    cov_m = xm_i * s_m**2
    cov_y = xy_i * s_y**2
    draws_m = rng.multivariate_normal(bm, cov_m, size=n_sims)
    draws_y = rng.multivariate_normal(by, cov_y, size=n_sims)

    a_d = draws_m[:, -1]
    b_d = draws_y[:, -2]
    c_d = draws_y[:, -1]
    acme_d = a_d * b_d
    ade_d = c_d
    total_d = acme_d + ade_d
    prop_d = np.divide(
        acme_d,
        total_d,
        out=np.full_like(acme_d, np.nan),
        where=np.abs(total_d) > 1e-9,
    )

    def band(d: FloatArray) -> tuple[float, float]:
        lo, hi = np.percentile(d[np.isfinite(d)], [2.5, 97.5])
        return float(lo), float(hi)

    acme_lo, acme_hi = band(acme_d)
    ade_lo, ade_hi = band(ade_d)
    tot_lo, tot_hi = band(total_d)
    prop_ok = prop_d[np.isfinite(prop_d)]
    prop_lo, prop_hi = np.percentile(prop_ok, [2.5, 97.5]) if prop_ok.size > 4 else (np.nan, np.nan)

    return {
        "acme": float(np.mean(acme_d)),
        "acme_lo": acme_lo,
        "acme_hi": acme_hi,
        "ade": float(np.mean(ade_d)),
        "ade_lo": ade_lo,
        "ade_hi": ade_hi,
        "total": float(np.mean(total_d)),
        "total_lo": tot_lo,
        "total_hi": tot_hi,
        "prop_mediated": float(np.nanmean(prop_d)) if prop_ok.size else math.nan,
        "prop_lo": float(prop_lo),
        "prop_hi": float(prop_hi),
        "a_path": a_coef,
        "b_path": b_coef,
        "c_prime": c_prime,
        "n_obs": float(n),
        "n_sims": float(n_sims),
        "acme_significant": float(acme_lo > 0.0 or acme_hi < 0.0),
    }


def synth_mediation(
    n: int = 600,
    a_path: float = 0.6,
    b_path: float = 0.8,
    c_prime: float = 0.2,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Mediation panel: T → M (a), M → Y (b), T → Y direct (c')."""
    rng = np.random.default_rng(seed)
    t_ = rng.normal(0.0, 1.0, n)
    m_ = a_path * t_ + rng.normal(0.0, 0.8, n)
    y_ = c_prime * t_ + b_path * m_ + rng.normal(0.0, 1.0, n)
    return {
        "treat": t_,
        "mediator": m_,
        "outcome": y_,
        "acme_true": np.array([a_path * b_path]),
        "ade_true": np.array([c_prime]),
    }


def bench_mediation_analysis(seed: int = 20261231 + 205) -> dict[str, float]:
    """Mediation self-check: ACME≈a·b recovered, interval covers truth,
    zero-mediation null has ACME interval covering 0. All ``synthetic_*``."""
    d = synth_mediation(seed=seed, a_path=0.6, b_path=0.8, c_prime=0.2)
    out = mediation_analysis(
        np.asarray(d["treat"]),
        np.asarray(d["mediator"]),
        np.asarray(d["outcome"]),
        n_sims=400,
        seed=seed + 1,
    )
    acme_true = float(d["acme_true"][0])
    covers = float(out["acme_lo"] <= acme_true <= out["acme_hi"])

    d0 = synth_mediation(seed=seed + 2, a_path=0.0)
    out0 = mediation_analysis(
        np.asarray(d0["treat"]),
        np.asarray(d0["mediator"]),
        np.asarray(d0["outcome"]),
        n_sims=400,
        seed=seed + 3,
    )
    # honest null readout: point estimate small relative to the real
    # ACME — interval coverage is reported, not gated (a-path false
    # positives occur at the nominal rate).
    null_acme = float(abs(float(out0["acme"])))
    null_covers0 = float(out0["acme_lo"] <= 0.0 <= out0["acme_hi"])

    out_b = mediation_analysis(
        np.asarray(d["treat"]),
        np.asarray(d["mediator"]),
        np.asarray(d["outcome"]),
        n_sims=400,
        seed=seed + 1,
    )

    return {
        "synthetic_acme": float(out["acme"]),
        "synthetic_acme_err": float(abs(float(out["acme"]) - acme_true)),
        "synthetic_acme_covers_true": covers,
        "synthetic_ade": float(out["ade"]),
        "synthetic_ade_err": float(abs(float(out["ade"]) - 0.2)),
        "synthetic_null_acme": null_acme,
        "synthetic_null_acme_covers0": null_covers0,
        "synthetic_detects": float(
            covers == 1.0
            and float(out["acme"]) > 0.2
            and null_acme < float(out["acme"]) * 0.35
            and abs(float(out["ade"]) - 0.2) < 0.3
        ),
        "synthetic_determinism": float(float(out["acme"]) == float(out_b["acme"])),
    }
