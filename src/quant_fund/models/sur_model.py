"""Zellner seemingly unrelated regressions (SUR / FGLS system).

Equations with correlated errors gain efficiency from joint
estimation: β_SUR = (X' (Σ⁻¹ ⊗ I) X)⁻¹ X' (Σ⁻¹ ⊗ I) y exploits
cross-equation residual covariance Σ. When Σ is diagonal or
regressors identical, SUR collapses to OLS — the gain statistic
quantifies the improvement.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure parameter recovery and
efficiency gains on generated systems — never market evidence.

References:
- Zellner, A. (1962). An efficient method of estimating
  seemingly unrelated regressions and tests for aggregation
  bias. *JASA* 57, 348-368.
- Zellner, A. (1963). Estimators for seemingly unrelated
  regression equations: some exact finite sample results.
  *JASA* 58 — when SUR ≡ OLS.
- Greene, W. H. (2012). *Econometric Analysis*, 7th ed.,
  ch. 10 — FGLS system estimation and iterated SUR.
- Srivastava, V. K., Giles, D. E. A. (1987). *Seemingly
  Unrelated Regression Equations Models*.

Composition: pure numpy — per-equation OLS residuals → Σ̂ →
Kronecker GLS, iterated once; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def sur_fit(
    y: list[FloatArray],
    x: list[FloatArray],
    n_iter: int = 1,
) -> dict[str, float]:
    """Estimate an m-equation SUR system by feasible GLS.

    ``y[i]`` is (T,), ``x[i]`` is (T × k_i). Returns per-equation
    coefficient vectors (raveled), residual correlation, and the
    efficiency gain over equation-by-equation OLS."""
    m = len(y)
    if m < 2 or m > 6:
        raise ValueError("need 2..6 equations")
    if len(x) != m:
        raise ValueError("y and x lists must match")
    ys = [np.asarray(v, dtype=np.float64).ravel() for v in y]
    xs = [np.asarray(v, dtype=np.float64) for v in x]
    t = ys[0].size
    if t < 40:
        raise ValueError("T>=40")
    if any(v.size != t for v in ys):
        raise ValueError("all y must share T")
    ws = []
    for i in range(m):
        xi = xs[i]
        if xi.ndim != 2 or xi.shape[0] != t:
            raise ValueError("x[i] must be (T × k_i)")
        ki = xi.shape[1]
        if ki < 1 or ki > 8 or ki >= t // 4:
            raise ValueError("k_i in 1..8, < T/4")
        if not np.all(np.isfinite(xi)) or not np.all(np.isfinite(ys[i])):
            raise ValueError("finite inputs required")
        if np.min(np.std(xi, axis=0)) < 1e-9:
            raise ValueError("each regressor must vary")
        ws.append(np.column_stack([np.ones(t), xi]))

    # equation-by-equation OLS (reference + residuals for Σ̂)
    b_ols = [np.linalg.lstsq(ws[i], ys[i], rcond=None)[0] for i in range(m)]
    e = np.column_stack([ys[i] - ws[i] @ b_ols[i] for i in range(m)])
    sigma_hat = (e.T @ e) / t
    if np.min(np.diag(sigma_hat)) < 1e-12:
        raise ValueError("degenerate residuals")

    def gls(sigma: FloatArray) -> list[FloatArray]:
        si = np.linalg.inv(sigma)
        # stacked SUR: X* = blockdiag(W_i), V = Σ ⊗ I_T →
        # β = (X*' (Σ⁻¹ ⊗ I) X*)⁻¹ X*' (Σ⁻¹ ⊗ I) y
        p = sum(w.shape[1] for w in ws)
        xt = np.zeros((m * t, p))
        col = 0
        for i in range(m):
            ki = ws[i].shape[1]
            xt[i * t : (i + 1) * t, col : col + ki] = ws[i]
            col += ki
        yv = np.concatenate(ys)
        sii = np.kron(si, np.eye(t))
        a = xt.T @ sii @ xt
        rhs = xt.T @ sii @ yv
        beta = np.linalg.solve(a + 1e-9 * np.eye(p), rhs)
        out = []
        col = 0
        for i in range(m):
            ki = ws[i].shape[1]
            out.append(beta[col : col + ki])
            col += ki
        return out

    beta = gls(sigma_hat)
    for _ in range(max(0, n_iter - 1)):
        e = np.column_stack([ys[i] - ws[i] @ beta[i] for i in range(m)])
        sigma_hat = (e.T @ e) / t
        beta = gls(sigma_hat)

    resid = np.column_stack([ys[i] - ws[i] @ beta[i] for i in range(m)])
    sigma_resid = (resid.T @ resid) / t
    d = np.sqrt(np.diag(sigma_resid))
    corr = float(sigma_resid[0, 1] / (d[0] * d[1]))

    # efficiency gain: trace of SUR vs OLS covariance (asymptotic
    # approx via sandwich of residual variance on design)
    def trace_var(betas: list[FloatArray]) -> float:
        tot = 0.0
        for i in range(m):
            xi = ws[i]
            vi = float(np.mean((ys[i] - xi @ betas[i]) ** 2))
            tot += float(np.trace(np.linalg.inv(xi.T @ xi))) * vi
        return tot

    var_ols = trace_var(b_ols)
    var_sur = trace_var(beta)

    return {
        "m": float(m),
        "t": float(t),
        "resid_corr": corr,
        "beta10": float(beta[0][1]),
        "beta20": float(beta[1][1]),
        "beta10_ols": float(b_ols[0][1]),
        "beta20_ols": float(b_ols[1][1]),
        "eff_gain": float(1.0 - var_sur / var_ols),
        "sigma11": float(sigma_resid[0, 0]),
        "sigma22": float(sigma_resid[1, 1]),
    }


def synth_sur(
    t: int = 400,
    beta1: float = 0.8,
    beta2: float = -0.5,
    rho: float = 0.7,
    seed: int = 0,
) -> dict[str, object]:
    """Two-equation SUR DGP with correlated errors ρ and
    different-but-overlapping regressors."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, t)
    x2 = rng.normal(0.0, 1.0, t)
    e1 = rng.normal(0.0, 1.0, t)
    e2 = rho * e1 + np.sqrt(max(1e-9, 1 - rho * rho)) * rng.normal(0.0, 1.0, t)
    y1 = 0.3 + beta1 * x1 + e1
    y2 = -0.2 + beta2 * x2 + e2
    return {
        "y": [y1, y2],
        "x": [x1.reshape(-1, 1), x2.reshape(-1, 1)],
        "beta1": np.array([beta1]),
        "beta2": np.array([beta2]),
        "rho": np.array([rho]),
    }


def bench_sur_model(seed: int = 20261231 + 229) -> dict[str, float]:
    """SUR self-check: coefficients recovered, residual correlation
    detected, efficiency gain vs OLS reported. All ``synthetic_*``."""
    d = synth_sur(seed=seed)
    out = sur_fit(d["y"], d["x"])  # type: ignore[arg-type]
    d0 = synth_sur(rho=0.0, seed=seed + 1)
    out0 = sur_fit(d0["y"], d0["x"])  # type: ignore[arg-type]
    out_b = sur_fit(d["y"], d["x"])  # type: ignore[arg-type]

    c = float(out["resid_corr"])
    return {
        "synthetic_beta1": float(out["beta10"]),
        "synthetic_beta1_err": float(abs(float(out["beta10"]) - 0.8)),
        "synthetic_beta2": float(out["beta20"]),
        "synthetic_beta2_err": float(abs(float(out["beta20"]) + 0.5)),
        "synthetic_resid_corr": c,
        "synthetic_corr_err": float(abs(c - 0.7)),
        "synthetic_corr_uncorr": float(out0["resid_corr"]),
        "synthetic_eff_gain": float(out["eff_gain"]),
        "synthetic_detects": float(
            abs(c - 0.7) < 0.2
            and abs(float(out["beta10"]) - 0.8) < 0.15
            and abs(float(out0["resid_corr"])) < 0.2
        ),
        "synthetic_determinism": float(
            c == float(out_b["resid_corr"]) and float(out["beta10"]) == float(out_b["beta10"])
        ),
    }
