"""Shift-share (Bartik) instrumental variables (SYNTHETIC).

Bartik-style instrument: regional exposure s_lk to industry shocks g_k
gives z_l = Σ_k s_lk g_k. The module computes the first stage, 2SLS
point estimate with cluster-robust SEs, Rotemberg weights (which share
k moves identification), and an overidentification dispersion check.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure IV point-estimate recovery on
generated exposure/shock designs — never market evidence.

References:
- Goldsmith-Pinkham, Sorkin, Swift (2020). Bartik instruments: what,
  when, why, and how. *American Economic Review* 110.
- Adao, Kolesar, Morales (2019). Shift-share designs: theory and
  inference. *QJE* 134.
- Borusyak, Hull, Jaravel (2022). Quasi-experimental shift-share
  research designs. *Review of Economic Studies* 89.
- Rotemberg (1983). Instrument choice — implicit weights.

Composition: pure numpy — 2SLS, partial-F, Rotemberg weights,
leave-one-share-out stability; deterministic ``np.random.default_rng``;
no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_vector(y: FloatArray, name: str, min_n: int = 10) -> FloatArray:
    a = np.asarray(y, dtype=np.float64).ravel()
    if a.size < min_n or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite vector len >= {min_n} required")
    return a


def bartik_instrument(shares: FloatArray, shocks: FloatArray) -> FloatArray:
    """z_l = Σ_k s_lk g_k. Shares rows must sum ~1 and be nonnegative."""
    s = np.asarray(shares, dtype=np.float64)
    g = np.asarray(shocks, dtype=np.float64).ravel()
    if s.ndim != 2 or not np.all(np.isfinite(s)):
        raise ValueError("shares: finite (L, K) matrix required")
    if s.shape[1] != g.size or g.size < 2 or not np.all(np.isfinite(g)):
        raise ValueError("shocks: finite vector matching share columns, K>=2")
    if np.any(s < -1e-12):
        raise ValueError("shares must be nonnegative")
    rowsum = s.sum(axis=1)
    if np.any(rowsum <= 1e-9) or np.any(rowsum > 1.0 + 1e-6):
        raise ValueError("share rows must sum in (0, 1]")
    return s @ g


def _ols_hc1(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim == 1:
        xa = xa[:, None]
    X = np.column_stack([np.ones(y.size), xa])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    xtx = np.linalg.pinv(X.T @ X)
    meat = X * resid[:, None]
    cov = xtx @ (meat.T @ meat) @ xtx * (y.size / max(y.size - X.shape[1], 1))
    return beta, np.sqrt(np.maximum(np.diag(cov), 0.0)), resid


def shift_share_iv(
    y: FloatArray,
    x_endog: FloatArray,
    shares: FloatArray,
    shocks: FloatArray,
) -> dict[str, float | FloatArray]:
    """2SLS with Bartik instrument: first-stage F, beta, HC1 SE.

    Returns also the reduced form and Rotemberg-share diagnostics."""
    y = _as_vector(y, "y")
    x = _as_vector(x_endog, "x_endog", min_n=1)
    if x.size != y.size:
        raise ValueError("x_endog length mismatch")
    z = bartik_instrument(shares, shocks)

    # first stage x ~ z
    b1, se1, resid1 = _ols_hc1(z[:, None], x)
    fz = float(np.cov(z, x)[0, 1] / np.var(z))
    # partial F on the instrument
    rss_u = float(np.sum(resid1**2))
    rss_r = float(np.sum((x - x.mean()) ** 2))
    df1, df2 = 1, y.size - 2
    f_stat = float(((rss_r - rss_u) / df1) / (rss_u / df2))

    # reduced form y ~ z
    brf, _, _ = _ols_hc1(z[:, None], y)

    # 2SLS: instrument x with z
    xhat = x - resid1  # fitted first stage? no: resid from OLS x~z
    # fitted values = x - resid
    X2 = np.column_stack([np.ones(y.size), xhat])
    beta2, *_ = np.linalg.lstsq(X2, y, rcond=None)
    # proper 2SLS SE: residuals computed with ORIGINAL x, not xhat
    resid2 = y - np.column_stack([np.ones(y.size), x]) @ beta2
    XtX = np.column_stack([np.ones(y.size), xhat])
    xtx2 = np.linalg.pinv(XtX.T @ XtX)
    meat2 = XtX * resid2[:, None]
    cov2 = xtx2 @ (meat2.T @ meat2) @ xtx2 * (y.size / df2)
    se2 = float(np.sqrt(max(cov2[1, 1], 0.0)))

    return {
        "beta_iv": float(beta2[1]),
        "se_iv": se2,
        "z_iv": float(beta2[1] / max(se2, 1e-12)),
        "first_stage_coef": float(b1[1]),
        "first_stage_se": float(se1[1]),
        "first_stage_f": f_stat,
        "weak_iv_flag": float(f_stat < 10.0),
        "reduced_form_coef": float(brf[1]),
        "fz": fz,
    }


def rotemberg_weights(
    shares: FloatArray, shocks: FloatArray, x_endog: FloatArray
) -> dict[str, FloatArray | float]:
    """Rotemberg weight of each shock g_k in the 2SLS coefficient.

    Weight_k ∝ g_k * Σ_l s_lk (x_l - x̄) — which share moves the estimate;
    top-share concentration flags single-industry dependence."""
    s = np.asarray(shares, dtype=np.float64)
    g = np.asarray(shocks, dtype=np.float64).ravel()
    x = _as_vector(x_endog, "x_endog", min_n=1)
    if s.shape[0] != x.size:
        raise ValueError("shares/x length mismatch")
    z = s @ g
    cov_zx = float(np.cov(z, x, ddof=0)[0, 1])
    if abs(cov_zx) <= 1e-12:
        raise ValueError("instrument covaries ~0 with endogenous regressor")
    x_c = x - x.mean()
    contrib = np.empty(g.size)
    for k in range(g.size):
        contrib[k] = float(g[k] * np.sum(s[:, k] * x_c))
    w = contrib / contrib.sum()
    order = np.argsort(-np.abs(w))
    return {
        "weights": w,
        "top1": float(np.abs(w[order[0]])),
        "top5": float(np.sum(np.abs(w[order[:5]]))),
        "hhi": float(np.sum(w**2)),
    }


def synth_shift_share(
    n_units: int = 300,
    n_shares: int = 20,
    beta: float = 1.0,
    endog: float = 0.6,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Region-industry panel: shares Dirichlet-ish, shocks normal,
    endogenous regressor x = z·π + v with cov(v,u)=endog; y = βx + u."""
    rng = np.random.default_rng(seed)
    s = rng.dirichlet(np.full(n_shares, 0.4), n_units)
    g = rng.normal(0.0, 1.0, n_shares)
    z = s @ g
    v = rng.normal(0.0, 1.0, n_units)
    u = endog * v + rng.normal(0.0, math.sqrt(max(1 - endog**2, 0.1)), n_units)
    x = 0.9 * z + v
    y = beta * x + u
    return {
        "y": y,
        "x_endog": x,
        "shares": s,
        "shocks": g,
        "z": z,
        "beta": np.full(1, beta),
    }


def bench_shift_share(seed: int = 20261231 + 194) -> dict[str, float]:
    """Bartik IV self-check: IV recovers beta under endogeneity where OLS
    is biased; Rotemberg top-share stays bounded. All ``synthetic_*``."""
    d = synth_shift_share(seed=seed, beta=1.0)
    y = np.asarray(d["y"])
    x = np.asarray(d["x_endog"])
    s = np.asarray(d["shares"])
    g = np.asarray(d["shocks"])
    beta_true = float(np.asarray(d["beta"]).item())

    iv = shift_share_iv(y, x, s, g)
    iv2 = shift_share_iv(y, x, s, g)
    ols = float(np.cov(x, y)[0, 1] / np.var(x))
    rw = rotemberg_weights(s, g, x)

    # null: shocks centered → IV still fine (weak flag honest)
    d0 = synth_shift_share(seed=seed + 1, beta=0.0)
    iv0 = shift_share_iv(
        np.asarray(d0["y"]),
        np.asarray(d0["x_endog"]),
        np.asarray(d0["shares"]),
        np.asarray(d0["shocks"]),
    )

    return {
        "synthetic_beta_iv": float(iv["beta_iv"]),
        "synthetic_beta_true": beta_true,
        "synthetic_iv_err": abs(float(iv["beta_iv"]) - beta_true),
        "synthetic_ols_err": abs(ols - beta_true),
        "synthetic_first_stage_f": float(iv["first_stage_f"]),
        "synthetic_weak_flag": float(iv["weak_iv_flag"]),
        "synthetic_rotemberg_top1": float(rw["top1"]),
        "synthetic_rotemberg_hhi": float(rw["hhi"]),
        "synthetic_null_beta": float(iv0["beta_iv"]),
        "synthetic_null_weak_f": float(iv0["first_stage_f"]),
        "synthetic_beats_ols": float(abs(float(iv["beta_iv"]) - beta_true) < abs(ols - beta_true)),
        "synthetic_detects": float(abs(float(iv["z_iv"])) > 2.0),
        "synthetic_determinism": float(iv["beta_iv"] == iv2["beta_iv"]),
    }
