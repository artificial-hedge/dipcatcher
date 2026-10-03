"""Sieve estimation — semiparametric partial-linear regression.

y = xβ + g(z) + ε where g is an unknown smooth function. Chen's
sieve approach approximates g by a finite basis expansion
(B-splines on a knot grid) and estimates (β, g) jointly by least
squares. β̂ is √n-consistent even though g is infinite-dimensional;
the linear-only fit is inconsistent whenever g is nonlinear and
correlated with x.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure β recovery and g fidelity on
generated partial-linear surfaces — never market evidence.

References:
- Chen, X. (2007). Large sample sieve estimation of semi-nonparametric
  models. In *Handbook of Econometrics* vol. 6B, 5549-5632 — sieve
  consistency and convergence rates.
- Robinson, P. M. (1988). Root-N-consistent semiparametric regression.
  *Econometrica* 56, 931-954 — the partial-linear model.
- de Boor, C. (1978). *A Practical Guide to Splines* — the B-spline
  basis built here (order-3, interior knots at quantiles).
- Wahba, G. (1990). *Spline Models for Observational Data* —
  smoothing-basis interpretation.

Composition: pure numpy — quantile-knot cubic B-spline basis via
Cox-de Boor recursion, joint OLS, basis purge to avoid collinearity;
deterministic ``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _bspline_basis(z: FloatArray, n_basis: int = 9, degree: int = 3) -> FloatArray:
    """Cubic B-spline basis on quantile interior knots (Cox-de Boor)."""
    zz = np.asarray(z, dtype=np.float64).ravel()
    lo, hi = float(zz.min()), float(zz.max())
    n_interior = n_basis - degree - 1
    if n_interior > 0:
        interior = np.quantile(zz, np.linspace(0, 1, n_interior + 2)[1:-1])
    else:
        interior = np.empty(0)
    knots = np.concatenate([np.repeat(lo, degree + 1), interior, np.repeat(hi, degree + 1)])
    n_b = knots.size - degree - 1
    b = np.zeros((zz.size, n_b))
    for i in range(n_b):
        b[:, i] = (zz >= knots[i]) & (zz < knots[i + 1])
    for d in range(1, degree + 1):
        nb = np.zeros((zz.size, n_b))
        for i in range(n_b - d):
            left = (zz - knots[i]) / max(knots[i + d] - knots[i], 1e-12) * b[:, i]
            right = (
                (knots[i + d + 1] - zz) / max(knots[i + d + 1] - knots[i + 1], 1e-12) * b[:, i + 1]
            )
            nb[:, i] = left + right
        b = nb
    b[:, -1] += zz == hi
    return b


def sieve_partial_linear(
    y: FloatArray,
    x: FloatArray,
    z: FloatArray,
    n_basis: int = 9,
) -> tuple[dict[str, float], FloatArray]:
    """Partial-linear sieve LS: y ~ x + basis(z).

    Joint OLS on [x, B(z)]; returns β plus the g-fit correlation
    and a linear-model reference."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    xx = np.atleast_2d(np.asarray(x, dtype=np.float64))
    zz = np.asarray(z, dtype=np.float64).ravel()
    if xx.shape[0] != yy.size:
        xx = xx.T
    n = yy.size
    if n < 80 or xx.shape[0] != n or zz.size != n:
        raise ValueError("x/z rows must equal len(y), n>=80")
    if not (np.all(np.isfinite(yy)) and np.all(np.isfinite(xx)) and np.all(np.isfinite(zz))):
        raise ValueError("finite inputs required")
    if np.ptp(zz) < 1e-9:
        raise ValueError("z must vary")
    if not (6 <= n_basis <= 20):
        raise ValueError("n_basis in 6..20")

    basis = _bspline_basis(zz, n_basis=n_basis)
    basis = basis[:, ~np.all(basis < 1e-10, axis=0)]
    design = np.column_stack([np.ones(n), xx, basis])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("design collinear — reduce n_basis")
    coef = np.linalg.lstsq(design, yy, rcond=None)[0]
    k = xx.shape[1]
    beta = coef[1 : 1 + k]
    g_hat = basis @ coef[1 + k :]
    g_hat = g_hat - g_hat.mean()
    resid = yy - design @ coef
    r2 = 1.0 - float(np.var(resid)) / float(np.var(yy))

    # linear reference
    d2 = np.column_stack([np.ones(n), xx, zz])
    b_lin = np.linalg.lstsq(d2, yy, rcond=None)[0]
    r2_lin = 1.0 - float(np.var(yy - d2 @ b_lin)) / float(np.var(yy))

    return {
        "n": float(n),
        "r2_sieve": r2,
        "r2_linear": r2_lin,
        **{f"beta_{j}": float(v) for j, v in enumerate(beta)},
        "beta_linear_0": float(b_lin[1]),
    }, g_hat


def synth_partial_linear(
    n: int = 900,
    beta: float = 0.8,
    z_corr: float = 0.6,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Partial-linear DGP: g(z)=sin(2z), x correlated with z so the
    linear fit is biased; heteroskedastic noise."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0, 4, n)
    x = z_corr * z / 4 + np.sqrt(1 - z_corr**2) * rng.normal(0, 1, n)
    g = np.sin(2 * z)
    y = beta * x + g + rng.normal(0, 0.15, n) * (0.6 + 0.4 * np.abs(x))
    return {"y": y, "x": x.reshape(-1, 1), "z": z, "g_true": g}


def bench_sieve_estimation(
    seed: int = 20261231 + 222,
) -> dict[str, float]:
    """Sieve self-check: β unbiased under nonlinear g correlated
    with x; ĝ tracks the true curve; linear reference is biased.
    All ``synthetic_*``."""
    d = synth_partial_linear(beta=0.8, z_corr=0.6, seed=seed)
    (out, g_hat) = sieve_partial_linear(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"]))
    g_true = np.asarray(d["g_true"]) - np.mean(np.asarray(d["g_true"]))
    g_cor = float(np.corrcoef(g_hat, g_true)[0, 1])
    g_rmse = float(np.sqrt(np.mean((g_hat - g_true) ** 2)))
    d0 = synth_partial_linear(beta=0.0, seed=seed + 1)
    (out0, _) = sieve_partial_linear(np.asarray(d0["y"]), np.asarray(d0["x"]), np.asarray(d0["z"]))
    (out_b, _) = sieve_partial_linear(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"]))

    b = float(out["beta_0"])
    return {
        "synthetic_beta": b,
        "synthetic_beta_err": float(abs(b - 0.8)),
        "synthetic_beta_linear": float(out["beta_linear_0"]),
        "synthetic_g_corr": g_cor,
        "synthetic_g_rmse": g_rmse,
        "synthetic_r2_sieve": float(out["r2_sieve"]),
        "synthetic_r2_linear": float(out["r2_linear"]),
        "synthetic_null_beta": float(abs(out0["beta_0"])),
        "synthetic_detects": float(abs(b - 0.8) < 0.15 and g_cor > 0.9),
        "synthetic_determinism": float(b == float(out_b["beta_0"])),
    }
