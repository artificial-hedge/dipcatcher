"""Marginal treatment effects via local instrumental variables.

References
----------
- Heckman, J.J. & Vytlacil, E. (2005). "Structural Equations, Treatment
  Effects, and Econometric Policy Evaluation." *Econometrica* 73(3),
  669-738.
- Heckman, J.J., Urzua, S. & Vytlacil, E. (2006). "Understanding
  Instrumental Variables in Models with Essential Heterogeneity."
  *Review of Economics and Statistics* 88(3), 389-432.
- Heckman, J.J. & Vytlacil, E. (2007). "Econometric Evaluation of
  Social Programs, Part II." *Handbook of Econometrics* 6B.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
Under the generalized Roy model D = 1[p(Z) >= U_D] with propensity
p(Z), the marginal treatment effect at unobserved-resistance u is

    MTE(u) = d/dp E[Y | p(Z) = p] |_{p = u}

— the derivative of the conditional mean of Y in the estimated
propensity (local IV). ATE = int_0^1 MTE(u) du; ATT and ATU follow the
standard weighting formulas omega_ATT(u) ∝ P(p>=u|D=1). Under a jointly
normal selection shock the synth's true MTE(u) = beta + rho *
Phi^{-1}(u) is monotone in u, and the local-linear derivative estimate
tracks it while a flat-Oaxaca constant effect cannot.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _design(x: FloatArray) -> FloatArray:
    n = x.shape[0]
    return np.column_stack([np.ones(n), x])


def _ols(y: FloatArray, x: FloatArray) -> FloatArray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _irls_logit(y: FloatArray, x: FloatArray, iters: int = 50) -> FloatArray:
    xx = _design(x)
    beta = np.zeros(xx.shape[1])
    lam = 1e-6
    for _ in range(iters):
        z = xx @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.maximum(p * (1.0 - p), 1e-9)
        h = xx.T @ (xx * w[:, None]) + lam * np.eye(xx.shape[1])
        g = xx.T @ (y - p) - lam * beta
        step = np.linalg.solve(h, g)
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return beta


def _local_linear_mean(y: FloatArray, p: FloatArray, grid: FloatArray, bw: float) -> FloatArray:
    """Local-linear regression of y on p at grid points."""
    out = np.zeros(grid.shape[0])
    for i, g0 in enumerate(grid):
        w = np.exp(-0.5 * ((p - g0) / bw) ** 2)
        sw = w.sum()
        if sw < 1e-9:
            out[i] = np.nan
            continue
        pm = p - g0
        s0 = sw
        s1 = float(w @ pm)
        s2 = float(w @ (pm * pm))
        t0 = float(w @ y)
        t1 = float(w @ (pm * y))
        det = s0 * s2 - s1 * s1
        if abs(det) < 1e-12:
            out[i] = t0 / s0
        else:
            out[i] = (s2 * t0 - s1 * t1) / det
    return out


def _local_linear_slope(y: FloatArray, p: FloatArray, grid: FloatArray, bw: float) -> FloatArray:
    """Slope of the local-linear fit at grid points — the MTE."""
    out = np.zeros(grid.shape[0])
    for i, g0 in enumerate(grid):
        w = np.exp(-0.5 * ((p - g0) / bw) ** 2)
        sw = w.sum()
        if sw < 1e-9:
            out[i] = np.nan
            continue
        pm = p - g0
        s0 = sw
        s1 = float(w @ pm)
        s2 = float(w @ (pm * pm))
        t0 = float(w @ y)
        t1 = float(w @ (pm * y))
        det = s0 * s2 - s1 * s1
        if abs(det) < 1e-12:
            out[i] = np.nan
        else:
            out[i] = (s0 * t1 - s1 * t0) / det
    return out


def marginal_te(
    y: FloatArray,
    treat: FloatArray,
    z: FloatArray,
    x: FloatArray | None = None,
    bw: float = 0.15,
    n_grid: int = 15,
) -> dict[str, float]:
    """Local-IV marginal treatment effect curve.

    ``z`` is a valid instrument (continuous preferred); the propensity is
    an IRLS logit on (x, z). MTE(u) is the local-linear slope of E[Y|p]
    evaluated on an interior grid, integrated for the ATE and
    ATT/ATU-weighted summaries.
    """
    yy = np.asarray(y, dtype=np.float64)
    tt = np.asarray(treat, dtype=np.float64)
    zz = np.asarray(z, dtype=np.float64)
    if zz.ndim == 1:
        zz = zz[:, None]
    if x is None:
        xx = np.zeros((yy.shape[0], 1), dtype=np.float64)
    else:
        xx = np.asarray(x, dtype=np.float64)
        if xx.ndim == 1:
            xx = xx[:, None]
    n = yy.shape[0]
    if yy.ndim != 1 or tt.ndim != 1 or zz.shape[0] != n or xx.shape[0] != n:
        raise ValueError("y, treat, z, x must share n rows")
    if n < 100:
        raise ValueError("need n >= 100")
    if (
        not np.all(np.isfinite(yy))
        or not np.all(np.isfinite(tt))
        or not np.all(np.isfinite(zz))
        or not np.all(np.isfinite(xx))
    ):
        raise ValueError("non-finite inputs")
    if not np.all((tt == 0.0) | (tt == 1.0)):
        raise ValueError("treat must be 0/1")

    px = np.column_stack([xx, zz])
    gb = _irls_logit(tt, px)
    p = np.clip(1.0 / (1.0 + np.exp(-_design(px) @ gb)), 1e-4, 1.0 - 1e-4)

    # Partial out observed X before the local-IV derivative: MTE(u) is
    # dE[Y - X'b | P = u]/du, so the x-channel through selection does
    # not contaminate the curve.
    y_r = yy - np.column_stack([np.ones(n), xx]) @ _ols(yy, np.column_stack([np.ones(n), xx]))

    lo, hi = np.quantile(p, [0.05, 0.95])
    grid = np.linspace(float(lo), float(hi), n_grid)
    mte = _local_linear_slope(y_r, p, grid, bw)
    ok = np.isfinite(mte)
    if ok.sum() < 4:
        raise ValueError("MTE curve degenerate")
    mte_g = mte[ok]
    gr = grid[ok]

    ate = float(np.trapezoid(mte_g, gr) / (gr[-1] - gr[0]))
    # ATT weight: density of p among treated at u (selection >= u).
    p_t = p[tt == 1]
    if p_t.shape[0] < 10:
        raise ValueError("too few treated")
    k_att = np.exp(-0.5 * ((gr[None, :] - p_t[:, None]) / bw) ** 2).mean(axis=0)
    w_att = k_att * (gr > 0)
    w_att = w_att / w_att.sum() if w_att.sum() > 0 else np.ones_like(gr) / gr.shape[0]
    att = float(mte_g @ w_att)

    return {
        "ate_mte": ate,
        "att_mte": att,
        "mte_mean": float(np.mean(mte_g)),
        "mte_slope_sd": float(np.std(mte_g)),
        "mte_min": float(np.min(mte_g)),
        "mte_max": float(np.max(mte_g)),
        "p_sd": float(np.std(p)),
        "naive_diff": float(np.mean(yy[tt == 1]) - np.mean(yy[tt == 0])),
    }


def synth_mte(
    n: int = 2000,
    seed: int = 20261231 + 280,
    rho: float = 0.7,
    beta: float = 0.5,
) -> dict[str, FloatArray]:
    """Generalized Roy synth: essential heterogeneity in treatment effect.

    V ~ N(0,1) drives selection: D = 1[0.8*z - V > 0]. Outcome shock
    U = rho*V + indep; Y = beta*D + 0.5*x + U + eps gives
    MTE(u) = beta + rho * Phi^{-1}(u): marginal entrants (high u) have
    systematically different gains — the signature the MTE curve
    detects.
    """
    rng = np.random.default_rng(seed)
    if n < 100:
        raise ValueError("n too small")
    z = rng.normal(0.0, 1.0, n)
    x = rng.normal(0.0, 1.0, n)
    v = rng.normal(0.0, 1.0, n)
    u = rho * v + np.sqrt(1.0 - rho**2) * rng.normal(0.0, 1.0, n)
    d = (0.8 * z + 0.2 * x - v > 0).astype(np.float64)
    y = beta * d + 0.5 * x + u + rng.normal(0.0, 0.3, n)
    return {
        "y": y,
        "treat": d,
        "z": z,
        "x": x,
        "true_beta": np.full(n, beta),
    }


def bench_marginal_treatment(seed: int = 20261231 + 280) -> dict[str, float]:
    """Wave-48 self-check: MTE-integrated ATE near truth while the naive
    difference is selection-biased."""
    d = synth_mte(seed=seed)
    a = marginal_te(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["z"]),
        np.asarray(d["x"]),
    )
    b = marginal_te(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["z"]),
        np.asarray(d["x"]),
    )
    detects = float(abs(a["ate_mte"] - 0.5) < abs(a["naive_diff"] - 0.5) * 0.5)
    return {
        "synthetic_detects": detects,
        "synthetic_determinism": float(a == b),
        "synthetic_ate_mte": a["ate_mte"],
        "synthetic_att_mte": a["att_mte"],
        "synthetic_mte_sd": a["mte_slope_sd"],
        "synthetic_mte_min": a["mte_min"],
        "synthetic_mte_max": a["mte_max"],
        "synthetic_naive_diff": a["naive_diff"],
        "synthetic_p_sd": a["p_sd"],
    }
