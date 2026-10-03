"""Nonparametric instrumental variables via series estimation.

y = g(x) + e with x endogenous (E[e|x] ≠ 0) but instruments z
satisfying E[e|z] = 0. Approximate g by a basis p(x) and
project it onto the instrument space: the series-IV estimator
β̂ = (P̂'P)^{-1} P̂'y with P̂ the projection of basis functions
onto span q(z) recovers g consistently while OLS on p(x) is
biased by the endogeneity.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure function recovery under
endogeneity on generated data — never market evidence.

References:
- Newey, W. K., Powell, J. L. (2003). Instrumental variable
  estimation of nonparametric models. *Econometrica* 71,
  1565-1578 — series/basis IV consistency.
- Ai, C., Chen, X. (2003). Efficient estimation of models with
  conditional moment restrictions containing unknown functions.
  *Econometrica* 71 — sieve minimum distance.
- Darolles, S., Fan, Y., Florens, J.-P., Renault, E. (2011).
  Nonparametric instrumental regression. *Econometrica* 79 —
  Tikhonov regularization alternative.
- Horowitz, J. L. (2011). Applied nonparametric instrumental
  variables estimation. *Econometrica* 79 — rates and basis
  choice.

Composition: pure numpy — polynomial/hinge basis, ridge-
regularized two-stage projection; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _basis(v: FloatArray, n_basis: int) -> FloatArray:
    """Piecewise-linear hinge basis on quantile knots + constant."""
    knots = np.quantile(v, np.linspace(0.15, 0.85, n_basis - 1))
    cols = [np.ones(v.size)]
    for kn in knots:
        cols.append(np.maximum(0.0, v - kn))
    return np.column_stack(cols)


def npiv_fit(
    y: FloatArray,
    x: FloatArray,
    z: FloatArray,
    n_basis: int = 6,
    ridge: float = 1e-3,
) -> dict[str, float]:
    """Series IV estimate of g(x) under instrument z.

    Returns fit correlation of ĝ vs true-comparable signal on a
    grid, plus endogeneity diagnostics vs OLS-on-basis."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    xx = np.asarray(x, dtype=np.float64).ravel()
    zz = np.asarray(z, dtype=np.float64).ravel()
    n = yy.size
    if xx.size != n or zz.size != n:
        raise ValueError("y, x, z must share n")
    if n < 100:
        raise ValueError("n>=100")
    if not (4 <= n_basis <= 12):
        raise ValueError("n_basis in 4..12")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(xx)) or not np.all(np.isfinite(zz)):
        raise ValueError("finite inputs required")
    if np.std(xx) < 1e-9 or np.std(zz) < 1e-9:
        raise ValueError("x and z must vary")
    # relevance gate: instrument must correlate with x
    fz = np.linalg.lstsq(np.column_stack([np.ones(n), zz]), xx, rcond=None)[0]
    x_hat = np.column_stack([np.ones(n), zz]) @ fz
    r2_fs = float(np.corrcoef(x_hat, xx)[0, 1] ** 2)
    if r2_fs < 0.02:
        raise ValueError("instrument too weak (first-stage R² < .02)")

    px = _basis(xx, n_basis)  # approximating basis in x
    qz = _basis(zz, n_basis + 2)  # instrument basis (richer)
    # project each column of px onto span(qz)
    qq_inv = np.linalg.inv(qz.T @ qz + ridge * np.eye(qz.shape[1]))
    p_hat = qz @ qq_inv @ qz.T @ px
    # series IV: β = (P̂'P)⁻¹ P̂' y
    a = p_hat.T @ px + ridge * np.eye(px.shape[1])
    beta_iv = np.linalg.solve(a, p_hat.T @ yy)
    g_iv = px @ beta_iv

    # OLS-on-basis reference (endogeneity bias)
    beta_ols = np.linalg.lstsq(px, yy, rcond=None)[0]
    g_ols = px @ beta_ols

    # Durbin-Wu-Hausman endogeneity check: augment OLS regression
    # with the first-stage residual v̂; a nonzero loading flags
    # endogeneity (DWH control-function test).
    v_hat = xx - x_hat
    aug = np.column_stack([px, v_hat])
    b_aug = np.linalg.lstsq(aug, yy, rcond=None)[0]
    resid_aug = yy - aug @ b_aug
    s2 = float(resid_aug @ resid_aug) / max(1, n - aug.shape[1])
    xtx_inv = np.linalg.inv(aug.T @ aug)
    se_v = float(np.sqrt(s2 * xtx_inv[-1, -1]))
    dwh_t = float(b_aug[-1] / max(se_v, 1e-12))

    return {
        "n": float(n),
        "n_basis": float(n_basis),
        "first_stage_r2": r2_fs,
        "g_iv_var": float(np.var(g_iv)),
        "g_ols_var": float(np.var(g_ols)),
        "iv_ols_gap": float(np.sqrt(np.mean((g_iv - g_ols) ** 2))),
        "dwh_t": dwh_t,
    }


def synth_endogenous(
    n: int = 800,
    confound: float = 0.8,
    kind: str = "sine",
    seed: int = 0,
) -> dict[str, FloatArray]:
    """y = g(x) + e, x = z·π + v, corr(v, e) = confound — OLS on x
    is biased; instrument z is clean. ``kind`` in sine | quad."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, n)
    v = rng.normal(0.0, 1.0, n)
    x = 0.8 * z + 0.6 * v
    e = confound * v + np.sqrt(max(1e-9, 1 - confound * confound)) * rng.normal(0.0, 0.6, n)
    if kind == "sine":
        g = np.sin(1.5 * x)
    else:
        g = 0.5 * x**2 - 1.0
    y = g + e
    return {"y": y, "x": x, "z": z, "g": g}


def bench_nonparametric_iv(
    seed: int = 20261231 + 231,
) -> dict[str, float]:
    """Series-IV self-check: ĝ_IV tracks true g under endogeneity
    while OLS-on-basis is pulled by the confounder.
    All ``synthetic_*``."""
    d = synth_endogenous(confound=0.7, seed=seed)
    y = np.asarray(d["y"])
    x = np.asarray(d["x"])
    z = np.asarray(d["z"])
    g = np.asarray(d["g"])
    out = npiv_fit(y, x, z)
    # recompute curves for error vs true g
    px = _basis(x, 6)
    qz = _basis(z, 8)
    p_hat = qz @ np.linalg.inv(qz.T @ qz + 1e-3 * np.eye(8)) @ qz.T @ px
    b_iv = np.linalg.solve(p_hat.T @ px + 1e-3 * np.eye(px.shape[1]), p_hat.T @ y)
    err_iv = float(np.sqrt(np.mean((px @ b_iv - g) ** 2)))
    b_ols = np.linalg.lstsq(px, y, rcond=None)[0]
    err_ols = float(np.sqrt(np.mean((px @ b_ols - g) ** 2)))
    d0 = synth_endogenous(confound=0.0, seed=seed + 1)
    out0 = npiv_fit(np.asarray(d0["y"]), np.asarray(d0["x"]), np.asarray(d0["z"]))
    out_b = npiv_fit(y, x, z)

    return {
        "synthetic_err_iv": err_iv,
        "synthetic_err_ols": err_ols,
        "synthetic_iv_gain": err_ols - err_iv,
        "synthetic_dwh_t": float(out["dwh_t"]),
        "synthetic_dwh_t_exog": float(out0["dwh_t"]),
        "synthetic_first_stage_r2": float(out["first_stage_r2"]),
        "synthetic_gap_exog": float(out0["iv_ols_gap"]),
        "synthetic_detects": float(
            err_iv < err_ols
            and abs(float(out["dwh_t"])) > 2.0
            and abs(float(out0["dwh_t"])) < abs(float(out["dwh_t"]))
            and float(out["first_stage_r2"]) > 0.3
        ),
        "synthetic_determinism": float(float(out["iv_ols_gap"]) == float(out_b["iv_ols_gap"])),
    }
