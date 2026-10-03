"""Partially-linear semiparametric regression.

Model: ``y_i = x_i'β + g(z_i) + ε_i`` with ``g`` an unknown smooth
function of the confounder ``z``. The parametric part ``β`` is estimated
at the parametric √n rate while ``g`` is recovered nonparametrically.

Modules
-------
* ``kernel_smooth`` — Nadaraya–Watson regression used internally.
* ``robinson_pl`` — Robinson (1988) residual-on-residual estimator:
  subtract the conditional means ``E[Y|Z]`` and ``E[X|Z]``, then OLS on
  the residuals ``ỹ = y − m_y(z)`` vs ``x̃ = x − m_x(z)``; ``g`` is
  recovered as ``E[Y − X'β̂ | Z]`` by a second smoother pass.
* ``speckman_pl`` — Speckman (1988) differencing estimator: project out
  ``Z`` nonparametrically from both sides via smoother matrix ``S``:
  ``β̂ = (X'(I−S)'(I−S) X)⁻¹ X'(I−S)'(I−S) Y``.
* ``series_pl`` — polynomial-basis (Hermite) series estimator for the
  ``g`` component, adding basis columns to the OLS design — useful as a
  parametric-parallel benchmark.
* ``cv_bandwidth`` — leave-one-out cross-validation over the smoothing
  bandwidth minimizing the residual-on-residual MSE.
* ``synth_partial_linear`` — synthetic design ``g(z)=sin(2πz)``, so OLS
  that ignores ``g`` is biased when ``corr(x,z)`` is high.

Honesty contract
----------------
* SYNTHETIC benches only; no prices or trading claims.
* β̂ inference uses the standard residual-on-residual sandwich;
  first-step bandwidth choice affects finite-sample bias — the bench
  reports the recovery errors transparently.

Composition
-----------
* Sits between ``functional_linear.py`` (wave 30, function-valued
  predictors) and ``dml.py`` (high-dimensional orthogonalization) — the
  classical scalar semiparametric core.

References
----------
* Robinson, P.M. (1988), "Root-N-Consistent Semiparametric Regression",
  Econometrica 56:931–954.
* Speckman, P. (1988), "Kernel Smoothing in Partial Linear Models",
  Journal of the Royal Statistical Society B 50:413–436.
* Härdle, W., Liang, H., Gao, J. (2000), Partially Linear Models,
  Physica-Verlag.
* Yatchew, A. (2003), Semiparametric Regression for the Applied
  Econometrician, Cambridge University Press.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import linalg

FloatArray = np.ndarray


def _check(
    y: FloatArray, x: FloatArray, z: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    y = np.asarray(y, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    z = np.asarray(z, dtype=np.float64).ravel()
    if not (y.size == x.shape[0] == z.size):
        raise ValueError("y, x, z must share n")
    if y.size < 20:
        raise ValueError("n >= 20 required")
    for arr in (y, x, z):
        if not np.all(np.isfinite(arr)):
            raise ValueError("inputs must be finite")
    return y, x, z


def kernel_smooth(
    y: FloatArray, z: FloatArray, bw: float, z_grid: FloatArray | None = None
) -> FloatArray:
    """Nadaraya–Watson Gaussian smooth of y on z (evaluated on z_grid)."""
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    tgt = z if z_grid is None else np.asarray(z_grid, dtype=np.float64).ravel()
    if bw <= 0:
        raise ValueError("bw must be positive")
    d = (tgt[:, None] - z[None, :]) / bw
    wts = np.exp(-0.5 * d * d)
    den = wts.sum(axis=1)
    den = np.where(den <= 0, 1e-12, den)
    return np.asarray((wts @ y) / den)


def cv_bandwidth(
    y: FloatArray, x: FloatArray, z: FloatArray, grid: FloatArray | None = None
) -> dict[str, FloatArray | float]:
    """Leave-one-out CV over bandwidth for the Robinson residual fit."""
    y, x, z = _check(y, x, z)
    if grid is None:
        s = float(np.std(z))
        grid = np.logspace(-1, 0, 12) * max(s, 1e-3)
    grid = np.asarray(grid, dtype=np.float64).ravel()
    if grid.size < 2:
        raise ValueError("grid needs >=2 bandwidths")
    errs = np.empty(grid.size)
    for gi, bw in enumerate(grid):
        m_y = kernel_smooth(y, z, bw)
        r_y = y - m_y
        m_x = np.column_stack([kernel_smooth(x[:, j], z, bw) for j in range(x.shape[1])])
        r_x = x - m_x
        beta_v = np.asarray(linalg.lstsq(r_x, r_y)[0])
        resid = r_y - r_x @ beta_v
        errs[gi] = float(np.mean(resid**2))
    best = float(grid[int(np.argmin(errs))])
    return {"bw": best, "grid": grid, "mse": errs}


def robinson_pl(
    y: FloatArray,
    x: FloatArray,
    z: FloatArray,
    bw: float | None = None,
    bw_cv: bool = True,
) -> dict[str, FloatArray | float]:
    """Robinson (1988) semiparametric partially-linear estimator.

    Returns ``beta``, ``se`` (Eicker–Huber–White on the residuals),
    ``g_hat`` (evaluated at the observed z), ``bw``, ``resid``.
    """
    y, x, z = _check(y, x, z)
    if bw is None:
        bw = float(cv_bandwidth(y, x, z)["bw"]) if bw_cv else float(np.std(z)) * 0.3
    m_y = kernel_smooth(y, z, bw)
    r_y = y - m_y
    m_x = np.column_stack([kernel_smooth(x[:, j], z, bw) for j in range(x.shape[1])])
    r_x = x - m_x
    beta = np.asarray(linalg.lstsq(r_x, r_y)[0])
    resid = r_y - r_x @ beta
    # sandwich variance on the residual design
    meat = r_x.T @ (r_x * (resid[:, None] ** 2))
    bread = r_x.T @ r_x
    var = linalg.solve(
        bread.T @ bread, linalg.solve(bread, meat, assume_a="pos").T, assume_a="pos"
    ).T
    se = np.sqrt(np.maximum(np.diag(var) / y.size, 0.0))
    g_hat = kernel_smooth(y - x @ beta, z, bw)
    return {
        "beta": beta,
        "se": se,
        "g_hat": g_hat,
        "resid": resid,
        "bw": float(bw),
    }


def speckman_pl(
    y: FloatArray, x: FloatArray, z: FloatArray, bw: float | None = None
) -> dict[str, FloatArray | float]:
    """Speckman (1988) projection-differencing estimator via smoother S."""
    y, x, z = _check(y, x, z)
    if bw is None:
        bw = float(np.std(z)) * 0.25
    d = (z[:, None] - z[None, :]) / bw
    s = np.exp(-0.5 * d * d)
    s = s / np.maximum(s.sum(axis=1, keepdims=True), 1e-12)
    i = np.eye(y.size)
    ims = i - s
    xt = ims @ x
    yt = ims @ y
    beta = np.asarray(linalg.lstsq(xt.T @ xt, xt.T @ yt)[0])
    resid = y - x @ beta - s @ (y - x @ beta)
    return {"beta": beta, "g_hat": np.asarray(s @ (y - x @ beta)), "resid": resid, "bw": float(bw)}


def series_pl(
    y: FloatArray, x: FloatArray, z: FloatArray, n_basis: int = 8
) -> dict[str, FloatArray | float]:
    """Polynomial series estimator: y = xβ + Σ_j c_j φ_j(z) + ε."""
    y, x, z = _check(y, x, z)
    if n_basis < 1 or n_basis > 40:
        raise ValueError("n_basis in [1,40]")
    zs = (z - z.mean()) / (z.std() + 1e-12)
    phi = np.polynomial.hermite_e.hermevander(zs, n_basis - 1)
    design = np.column_stack([x, phi])
    coef = np.asarray(linalg.lstsq(design, y)[0])
    beta = coef[: x.shape[1]]
    g_coef = coef[x.shape[1] :]
    g_hat = phi @ g_coef
    resid = y - design @ coef
    return {
        "beta": beta,
        "g_hat": np.asarray(g_hat),
        "resid": resid,
        "n_basis": float(n_basis),
    }


def synth_partial_linear(
    n: int = 400, corr_xz: float = 0.6, seed: int = 0
) -> dict[str, FloatArray | np.float64]:
    """SYNTHETIC: y = 1 + 1.5x + sin(2πz) + ε; corr(x,z) controls OLS bias."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0, 1, n)
    v = rng.standard_normal(n)
    x = corr_xz * np.sin(2 * np.pi * z) + math.sqrt(max(1 - corr_xz**2, 0.05)) * v
    y = 1.0 + 1.5 * x + np.sin(2 * np.pi * z) + 0.3 * rng.standard_normal(n)
    return {
        "y": y,
        "x": x[:, None],
        "z": z,
        "g_true": np.sin(2 * np.pi * z),
        "beta_true": np.array([1.5]),
    }


def bench_partial_linear(seed: int = 20261231 + 175) -> dict[str, float]:
    """SYNTHETIC: semiparametric β recovery beats naive OLS when corr(x,g) high."""
    d = synth_partial_linear(n=400, corr_xz=0.6, seed=seed)
    y = np.asarray(d["y"])
    x = np.asarray(d["x"])
    z = np.asarray(d["z"])
    g_true = np.asarray(d["g_true"])
    fit = robinson_pl(y, x, z)
    sp = speckman_pl(y, x, z)
    se = series_pl(y, x, z, n_basis=9)
    beta = float(np.asarray(fit["beta"])[0])
    beta_sp = float(np.asarray(sp["beta"])[0])
    beta_se = float(np.asarray(se["beta"])[0])
    # naive OLS biased by omitted sin(2πz) since corr(x,sin)=0.6
    des = np.column_stack([np.ones(y.size), x])
    beta_ols = float(np.asarray(linalg.lstsq(des, y)[0])[1])
    g_hat = np.asarray(fit["g_hat"])
    order = np.argsort(z)
    g_corr = float(np.corrcoef(g_hat[order], g_true[order])[0, 1])
    e1 = robinson_pl(y, x, z, bw=float(fit["bw"]), bw_cv=False)
    return {
        "synthetic_beta_err": abs(beta - 1.5),
        "synthetic_beta_speckman_err": abs(beta_sp - 1.5),
        "synthetic_beta_series_err": abs(beta_se - 1.5),
        "synthetic_beta_ols_err": abs(beta_ols - 1.5),
        "synthetic_semi_beats_ols": float(abs(beta - 1.5) < abs(beta_ols - 1.5)),
        "synthetic_g_corr": g_corr,
        "synthetic_bw": float(fit["bw"]),
        "synthetic_determinism": float(
            np.allclose(np.asarray(fit["beta"]), np.asarray(e1["beta"]))
        ),
    }
