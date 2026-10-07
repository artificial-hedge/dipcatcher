"""Fama-MacBeth two-pass asset-pricing regression (SYNTHETIC).

Pass 1: per-asset time-series betas on factor returns. Pass 2:
cross-sectional regression of average returns on betas each period;
coefficient estimates are the time-series mean of the per-period
risk prices with Fama-MacBeth SEs (mean/√T), plus the Shanken (1992)
errors-in-variables correction for estimated betas.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure risk-price recovery on generated
factor/return panels — never market evidence.

References:
- Fama, MacBeth (1973). Risk, return, and equilibrium: empirical
  tests. *JPE* 81.
- Shanken (1992). On the estimation of beta-pricing models. *RFS* 5.
- Cochrane (2005). *Asset Pricing*, ch. 12-13 (two-pass machinery).
- Jagannathan, Skoulakis, Wang (2010). The analysis of the
  cross-section of security returns. *Handbook of Empirical
  Economics and Finance* 2.

Composition: pure numpy — cross-sectional OLS per period,
Shanken EIV factor adjustment; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as2(m: FloatArray, name: str) -> FloatArray:
    a = np.asarray(m, dtype=np.float64)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite 2-D matrix required")
    return a


def estimate_betas(returns: FloatArray, factors: FloatArray) -> dict[str, FloatArray | float]:
    """Pass 1: per-asset OLS betas on [1, factors]."""
    r = _as2(returns, "returns")
    f = _as2(factors, "factors")
    if r.shape[0] != f.shape[0]:
        raise ValueError("returns/factors T mismatch")
    t, n = r.shape
    k = f.shape[1]
    if n < k + 2 or t < k + 4:
        raise ValueError("need n>k+2 assets and t>k+4 periods")
    X = np.column_stack([np.ones(t), f])
    B, *_ = np.linalg.lstsq(X, r, rcond=None)  # (k+1, n)
    beta = B[1:].T  # (n, k)
    resid = r - X @ B
    sigma_eps = np.asarray(resid.var(axis=0, ddof=k + 1))
    return {
        "betas": beta,
        "resid_var": sigma_eps,
        "factor_cov": np.cov(f.T),
        "factor_means": f.mean(axis=0),
    }


def fama_macbeth(returns: FloatArray, factors: FloatArray) -> dict[str, float | FloatArray]:
    """Full two-pass: risk prices λ = mean_t γ_t, FM SE = std(γ_t)/√T.

    Shanken correction inflates SEs by (1 + λ̂'Σ_f^{-1}λ̂) for the EIV
    in estimated betas. Returns per-factor price, SEs, t-stats, and
    the cross-sectional R²."""
    r = _as2(returns, "returns")
    f = _as2(factors, "factors")
    t, n = r.shape
    k = f.shape[1]

    b = estimate_betas(r, f)
    beta = np.asarray(b["betas"])

    # pass 2: per-period cross-sectional OLS r_t ~ [1, beta]
    gammas = np.empty((t, k))
    lam0 = np.empty(t)
    r2 = np.empty(t)
    Xb = np.column_stack([np.ones(n), beta])
    pinv = np.linalg.pinv(Xb.T @ Xb) @ Xb.T
    for tt in range(t):
        g = pinv @ r[tt]
        lam0[tt] = g[0]
        gammas[tt] = g[1:]
        fit = Xb @ g
        ss_res = float(((r[tt] - fit) ** 2).sum())
        ss_tot = float(((r[tt] - r[tt].mean()) ** 2).sum())
        r2[tt] = 1.0 - ss_res / max(ss_tot, 1e-12)

    lam = gammas.mean(axis=0)
    se_fm = gammas.std(axis=0, ddof=1) / math.sqrt(t)
    # Shanken EIV correction: multiply var-cov by (1 + λ'Σ_f^{-1}λ)
    sigma_f = np.asarray(b["factor_cov"])
    sf_inv = np.linalg.pinv(sigma_f)
    shanken_factor = float(1.0 + lam @ sf_inv @ lam)
    se_shanken = se_fm * math.sqrt(max(shanken_factor, 0.0))

    lam0_se = float(lam0.std(ddof=1) / math.sqrt(t))
    return {
        "lambda": lam,
        "se_fm": se_fm,
        "se_shanken": se_shanken,
        "t_fm": lam / np.maximum(se_fm, 1e-12),
        "t_shanken": lam / np.maximum(se_shanken, 1e-12),
        "lambda0": float(lam0.mean()),
        "lambda0_se": lam0_se,
        "r2_mean": float(r2.mean()),
        "shanken_inflation": shanken_factor,
        "n_assets": float(n),
        "t_periods": float(t),
    }


def cross_section_r2_profile(returns: FloatArray, factors: FloatArray) -> FloatArray:
    """Per-period cross-sectional R² vector (diagnostic)."""
    r = _as2(returns, "returns")
    f = _as2(factors, "factors")
    b = estimate_betas(r, f)
    beta = np.asarray(b["betas"])
    t, n = r.shape
    Xb = np.column_stack([np.ones(n), beta])
    pinv = np.linalg.pinv(Xb.T @ Xb) @ Xb.T
    out = np.empty(t)
    for tt in range(t):
        g = pinv @ r[tt]
        fit = Xb @ g
        ss_res = float(((r[tt] - fit) ** 2).sum())
        ss_tot = float(((r[tt] - r[tt].mean()) ** 2).sum())
        out[tt] = 1.0 - ss_res / max(ss_tot, 1e-12)
    return out


def synth_factor_panel(
    n: int = 60,
    t: int = 120,
    k: int = 2,
    lam_true: tuple[float, ...] = (0.6, 0.3),
    beta_spread: float = 0.6,
    resid_sd: float = 1.0,
    priced_idio: float = 0.0,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Factor panel: r_it = a_i + b_i'f_t + eps; a_i = b_i'λ + priced_idio·ν_i.

    ``priced_idio`` adds a cross-sectional pricing error (tests FM SE
    honesty — SE should widen with priced idiosyncratic dispersion)."""
    rng = np.random.default_rng(seed)
    lam = np.asarray(lam_true)[:k]
    f = rng.normal(0.0, 1.0, (t, k))
    beta = rng.normal(0.0, beta_spread, (n, k))
    nu = rng.normal(0.0, 0.4, n)
    alpha = beta @ lam + priced_idio * nu
    eps = rng.normal(0.0, resid_sd, (t, n))
    r = alpha[None, :] + f @ beta.T + eps
    return {
        "returns": r,
        "factors": f,
        "lambda_true": lam,
        "betas": beta,
        "nu_priced": nu,
    }


def bench_fama_macbeth(seed: int = 20261231 + 199) -> dict[str, float]:
    """FM two-pass self-check: recovered λ close to true λ; Shanken
    inflates SEs; priced-idio noise widens dispersion. All ``synthetic_*``."""
    d = synth_factor_panel(seed=seed)
    fm = fama_macbeth(np.asarray(d["returns"]), np.asarray(d["factors"]))
    lam = np.asarray(fm["lambda"])
    lam_true = np.asarray(d["lambda_true"])
    err = float(np.abs(lam - lam_true).max())

    d2 = synth_factor_panel(seed=seed + 1, priced_idio=0.8)
    fm2 = fama_macbeth(np.asarray(d2["returns"]), np.asarray(d2["factors"]))

    d0 = synth_factor_panel(seed=seed + 2, lam_true=(0.0, 0.0))
    fm0 = fama_macbeth(np.asarray(d0["returns"]), np.asarray(d0["factors"]))

    fm_b = fama_macbeth(np.asarray(d["returns"]), np.asarray(d["factors"]))

    return {
        "synthetic_lambda_1": float(lam[0]),
        "synthetic_lambda_2": float(lam[1]),
        "synthetic_lambda_err": err,
        "synthetic_se_shanken_gt_fm": float(
            np.all(np.asarray(fm["se_shanken"]) >= np.asarray(fm["se_fm"]))
        ),
        "synthetic_shanken_inflation": float(fm["shanken_inflation"]),
        "synthetic_r2_mean": float(fm["r2_mean"]),
        "synthetic_priced_idio_se_ratio": float(
            np.asarray(fm2["se_fm"]).mean() / max(np.asarray(fm["se_fm"]).mean(), 1e-12)
        ),
        "synthetic_null_lambda": float(np.abs(np.asarray(fm0["lambda"])).max()),
        "synthetic_detects": float(err < 0.5 and float(fm["r2_mean"]) > 0.1),
        "synthetic_determinism": float(
            float(np.asarray(fm["lambda"])[0]) == float(np.asarray(fm_b["lambda"])[0])
        ),
    }
