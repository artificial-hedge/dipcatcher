"""Functional data analysis: FPCA and function-on-scalar regression.

Each observation is a curve sampled on a shared grid (e.g. a daily
intraday path, a term structure, a yield curve). Implements Ramsay &
Silverman machinery on B-spline bases: smoothed functional PCA by
covariance eigenanalysis (Ramsay & Silverman 2005 ch. 8), and a
function-on-scalar linear model estimated as a penalized multivariate
regression of the coefficient functions (Morris 2015 functional
regression review), with second-difference roughness penalization.

References
----------
- Ramsay & Silverman (2005). *Functional Data Analysis*, 2nd ed.,
  ch. 8-9 (FPCA), ch. 13-14 (functional linear models).
- Morris (2015). Functional regression. *Annual Review of Statistics*
  2:321-359.
- Goldsmith, Bobb, Crainiceanu, Caffo & Reich (2011). Penalized
  functional regression. *JCGS* 20(4).

Honesty
-------
All curves are SYNTHETIC (planted eigenfunctions/coefficient surfaces);
keys report reconstruction error, eigenfunction recovery via congruence,
and cross-validated fit — never claims about real curves.

Composition notes
-----------------
- ``models/factor_nowcast.py``: discrete-time factor models — FPCA is
  the function-space analogue (eigenfunctions instead of loadings).
- ``metrics/rqa.py`` (wave 29): scalar-series nonlinear analysis —
  this module handles curve-valued data instead.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_curves(x: FloatArray, min_n: int = 10, min_t: int = 8) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < min_n or x.shape[1] < min_t:
        raise ValueError("curves must be (n>=min_n, t>=min_t)")
    if not np.all(np.isfinite(x)):
        raise ValueError("curves must be finite")
    return x


def bspline_basis(grid: FloatArray, n_basis: int, degree: int = 3) -> FloatArray:
    """Cubic B-spline basis on a grid via the Cox-de Boor recursion,
    with open-uniform knots. Returns (len(grid), n_basis)."""
    grid = np.asarray(grid, dtype=np.float64).ravel()
    if grid.size < n_basis + degree + 1:
        raise ValueError("grid too short for basis size")
    lo, hi = float(grid.min()), float(grid.max())
    # open uniform knot vector
    n_int = n_basis - degree - 1
    interior = np.linspace(lo, hi, n_int + 2)[1:-1] if n_int > 0 else np.array([])
    knots = np.concatenate([np.full(degree + 1, lo), interior, np.full(degree + 1, hi)])
    n_knots = knots.size
    # Cox-de Boor: build B_{i,0} then iterate degree
    b = np.zeros((grid.size, n_knots - 1))
    for i in range(n_knots - 1):
        mask = (grid >= knots[i]) & (grid < knots[i + 1])
        if i == n_knots - 2:  # last span includes right endpoint
            mask |= grid == hi
        b[:, i] = mask
    for d in range(1, degree + 1):
        nb = np.zeros((grid.size, n_knots - 1 - d))
        for i in range(n_knots - 1 - d):
            d1 = knots[i + d] - knots[i]
            d2 = knots[i + d + 1] - knots[i + 1]
            t1 = (grid - knots[i]) / d1 if d1 > 0 else 0.0 * grid
            t2 = (knots[i + d + 1] - grid) / d2 if d2 > 0 else 0.0 * grid
            nb[:, i] = (t1 * b[:, i] if d1 > 0 else 0.0) + (t2 * b[:, i + 1] if d2 > 0 else 0.0)
        b = nb
    out = np.asarray(b[:, :n_basis])
    if out.shape[1] < n_basis:
        out = np.column_stack([out, np.zeros((grid.size, n_basis - out.shape[1]))])
    # partition of unity: B-splines sum to 1; the open-uniform right
    # endpoint convention leaves the last grid point's basis mass on
    # the final column.
    row_sums = out.sum(axis=1)
    for i in range(out.shape[0]):
        if row_sums[i] > 1e-12:
            out[i] /= row_sums[i]
        else:
            out[i, -1] = 1.0
    return np.asarray(out)


def fpca(
    curves: FloatArray,
    n_components: int = 3,
    n_basis: int | None = None,
    smooth_lambda: float = 1e-3,
) -> dict[str, FloatArray | np.float64]:
    """Functional PCA via B-spline-smoothed covariance eigenanalysis.

    Returns eigenfunctions evaluated on the input grid, eigenvalues,
    component scores, and per-curve reconstruction error at the chosen
    rank. Basis expansion of each curve precedes eigenanalysis of the
    covariance operator in coefficient space (James, Hastie & Sugar
    2000 mixed-effects-free variant).
    """
    curves = _as_curves(curves)
    n, t = curves.shape
    grid = np.linspace(0.0, 1.0, t)
    if n_basis is None:
        n_basis = min(15, t - 2)
    basis = bspline_basis(grid, n_basis)
    # project curves onto basis: coef_i = argmin ||x_i - B c||
    coef = np.linalg.lstsq(basis.T @ basis + 1e-6 * np.eye(n_basis), basis.T @ curves.T)[
        0
    ].T  # (n, n_basis)
    mean_coef = coef.mean(axis=0)
    c_cent = coef - mean_coef
    cov = c_cent.T @ c_cent / max(n - 1, 1)
    # roughness penalty on eigenfunctions via second-difference
    pen = np.zeros((n_basis, n_basis))
    for i in range(1, n_basis - 1):
        for off in (-1, 0, 1):
            pen[i, i] += 4.0 if off == 0 else -2.0
    cov_s = cov + smooth_lambda * pen
    eigval, eigvec = np.linalg.eigh(cov_s)
    order = np.argsort(eigval)[::-1]
    n_components = min(n_components, n_basis, eigval.size)
    vals = eigval[order][:n_components]
    vecs = eigvec[:, order][:, :n_components]
    eigenfuncs = basis @ vecs  # (t, n_components)
    scores = c_cent @ vecs  # (n, n_components)
    fitted = (mean_coef + scores @ vecs.T) @ basis.T
    rel_err = float(np.linalg.norm(curves - fitted) / np.linalg.norm(curves - curves.mean(axis=0)))
    return {
        "eigenfunctions": np.asarray(eigenfuncs),
        "eigenvalues": np.asarray(vals),
        "scores": np.asarray(scores),
        "fitted": np.asarray(fitted),
        "mean_curve": np.asarray(mean_coef @ basis.T),
        "rel_err": np.float64(rel_err),
        "variance_explained": np.asarray(vals / vals.sum().clip(min=1e-12)),
    }


def functional_lm(
    y_curves: FloatArray,
    x: FloatArray,
    n_basis: int | None = None,
    smooth_lambda: float = 1e-2,
) -> dict[str, FloatArray | np.float64]:
    """Function-on-scalar regression: y_i(t) = α(t) + β(t) x_i + ε_i(t).

    Fits the coefficient functions in a B-spline basis with a
    second-difference roughness penalty — the penalized concurrent
    functional linear model.
    """
    y = _as_curves(y_curves)
    xa = np.asarray(x, dtype=np.float64).ravel()
    n, t = y.shape
    if xa.size != n:
        raise ValueError("x must align with curve rows")
    if float(np.std(xa)) < 1e-12:
        raise ValueError("x must have nonzero variance")
    grid = np.linspace(0.0, 1.0, t)
    if n_basis is None:
        n_basis = min(12, t - 2)
    basis = bspline_basis(grid, n_basis)
    # 1) pointwise cross-sectional regression at each grid point —
    #    y_i(t) = a_t + b_t x_i, giving raw coefficient functions.
    des = np.column_stack([np.ones(n), xa])
    pinv_des = np.linalg.pinv(des)
    raw = pinv_des @ y  # (2, t)
    # 2) smooth each coefficient function in the B-spline basis with a
    #    second-difference roughness penalty (Ramsay-Silverman).
    d2 = np.diff(np.eye(n_basis), n=2, axis=0)
    pen = d2.T @ d2
    bt_b = basis.T @ basis
    sm = np.linalg.solve(bt_b + smooth_lambda * pen + 1e-8 * np.eye(n_basis), basis.T)
    alpha_fn = basis @ (sm @ raw[0])
    beta_fn = basis @ (sm @ raw[1])
    fitted = alpha_fn[None, :] + xa[:, None] * beta_fn[None, :]
    resid = y - fitted
    r2 = float(1.0 - np.sum(resid**2) / np.sum((y - y.mean(axis=0)) ** 2))
    return {
        "alpha": np.asarray(alpha_fn),
        "beta": np.asarray(beta_fn),
        "fitted": np.asarray(fitted),
        "r2": np.float64(r2),
        "grid": np.asarray(grid),
    }


def synth_fpca(n: int = 60, t: int = 50, rank: int = 2, seed: int = 0) -> dict[str, FloatArray]:
    """Curves generated from two sinusoidal eigenfunctions + noise."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, t)
    e1 = np.sin(2 * np.pi * grid)
    e2 = np.cos(4 * np.pi * grid)
    s1 = rng.standard_normal(n) * 2.0
    s2 = rng.standard_normal(n) * 1.0
    x = s1[:, None] * e1[None, :] + s2[:, None] * e2[None, :]
    x += 0.15 * rng.standard_normal((n, t))
    return {
        "X": np.asarray(x),
        "e1": np.asarray(e1),
        "e2": np.asarray(e2),
        "grid": np.asarray(grid),
    }


def synth_flm(n: int = 80, t: int = 40, seed: int = 0) -> dict[str, FloatArray]:
    """Function-on-scalar synth: β(t) = t(1-t) peaked coefficient."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, t)
    beta = 4.0 * grid * (1 - grid)
    alpha = np.sin(2 * np.pi * grid)
    x = rng.standard_normal(n)
    y = alpha[None, :] + x[:, None] * beta[None, :]
    y += 0.2 * rng.standard_normal((n, t))
    return {
        "Y": np.asarray(y),
        "x": np.asarray(x),
        "beta_true": np.asarray(beta),
        "alpha_true": np.asarray(alpha),
        "grid": np.asarray(grid),
    }


def bench_functional_linear(seed: int = 20261231 + 169) -> dict[str, float]:
    """SYNTHETIC FPCA eigenfunction recovery + FLM beta recovery."""
    fp = synth_fpca(n=80, t=60, seed=seed)
    out = fpca(fp["X"], n_components=2, n_basis=12)
    eigs = np.asarray(out["eigenfunctions"])

    # Tucker congruence against planted e1/e2
    def _cong(a: FloatArray, b: FloatArray) -> float:
        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()
        return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    c1 = max(_cong(fp["e1"], eigs[:, j]) for j in range(2))
    c2 = max(_cong(fp["e2"], eigs[:, j]) for j in range(2))
    fl = synth_flm(n=100, t=50, seed=seed + 1)
    fit = functional_lm(fl["Y"], fl["x"], n_basis=10, smooth_lambda=1e-2)
    beta_hat = np.asarray(fit["beta"])
    beta_true = fl["beta_true"]
    beta_relerr = float(np.linalg.norm(beta_hat - beta_true) / np.linalg.norm(beta_true))
    d1 = fpca(fp["X"], n_components=2, n_basis=12)["rel_err"]
    d2 = fpca(fp["X"], n_components=2, n_basis=12)["rel_err"]
    return {
        "synthetic_fpca_relerr": float(out["rel_err"]),
        "synthetic_eig1_congruence": c1,
        "synthetic_eig2_congruence": c2,
        "synthetic_var_explained_1": float(np.asarray(out["variance_explained"])[0]),
        "synthetic_flm_r2": float(fit["r2"]),
        "synthetic_beta_relerr": beta_relerr,
        "synthetic_determinism": float(d1 == d2),
    }
