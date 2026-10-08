"""Nonparametric kernel regression — Nadaraya-Watson and local linear (SYNTHETIC).

m(x) = E[y|x] estimated by locally-weighted averaging. Local linear
estimates the conditional mean AND its derivative with reduced
boundary bias; Nadaraya-Watson (local constant) is the simpler
variant whose boundary bias motivates the linear upgrade.
Leave-one-out CV selects the bandwidth.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure regression-curve recovery on
generated nonlinear panels — never market evidence.

References:
- Nadaraya, E. A. (1964). On estimating regression. *Theory of
  Probability and its Applications* 9, 141-142.
- Watson, G. S. (1964). Smooth regression analysis. *Sankhyā* A26,
  359-372 — the local-constant estimator.
- Fan, J., Gijbels, I. (1996). *Local Polynomial Modelling and Its
  Applications*. Chapman & Hall — local-linear bias properties.
- Li, Q., Racine, J. S. (2007). *Nonparametric Econometrics*.
  Princeton UP — least-squares CV bandwidth selection.
- Härdle, W. (1990). *Applied Nonparametric Regression*. Cambridge.

Composition: pure numpy — Gaussian kernel, LOO-CV bandwidth grid,
derivative estimate from the local-linear slope; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_xy(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    xx = np.asarray(x, dtype=np.float64).ravel()
    yy = np.asarray(y, dtype=np.float64).ravel()
    if xx.size != yy.size or xx.size < 30:
        raise ValueError("equal-length x and y, n>=30")
    if not np.all(np.isfinite(xx)) or not np.all(np.isfinite(yy)):
        raise ValueError("finite x and y required")
    return xx, yy


def _gauss(u: FloatArray) -> FloatArray:
    return np.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)


def nadaraya_watson(
    x: FloatArray,
    y: FloatArray,
    x_eval: FloatArray,
    bandwidth: float,
) -> FloatArray:
    """Local-constant kernel smoother evaluated at ``x_eval``."""
    xx, yy = _as_xy(x, y)
    xe = np.asarray(x_eval, dtype=np.float64).ravel()
    if bandwidth <= 0:
        raise ValueError("bandwidth must be > 0")
    out = np.zeros(xe.size)
    for i, xv in enumerate(xe):
        w = _gauss((xx - xv) / bandwidth)
        s = w.sum()
        if s < 1e-10:
            raise ValueError("evaluation point too far from data")
        out[i] = float(w @ yy / s)
    return out


def local_linear(
    x: FloatArray,
    y: FloatArray,
    x_eval: FloatArray,
    bandwidth: float,
) -> tuple[FloatArray, FloatArray]:
    """Local-linear smoother: returns (fit, derivative) at ``x_eval``."""
    xx, yy = _as_xy(x, y)
    xe = np.asarray(x_eval, dtype=np.float64).ravel()
    if bandwidth <= 0:
        raise ValueError("bandwidth must be > 0")
    fit = np.zeros(xe.size)
    deriv = np.zeros(xe.size)
    for i, xv in enumerate(xe):
        w = _gauss((xx - xv) / bandwidth)
        if w.sum() < 1e-10:
            raise ValueError("evaluation point too far from data")
        dx = xx - xv
        s0 = w.sum()
        s1 = float((w * dx).sum())
        s2 = float((w * dx * dx).sum())
        t0 = float(w @ yy)
        t1 = float((w * dx) @ yy)
        det = s0 * s2 - s1 * s1
        if abs(det) < 1e-12:
            fit[i] = t0 / s0
            deriv[i] = 0.0
            continue
        fit[i] = (s2 * t0 - s1 * t1) / det
        deriv[i] = (s0 * t1 - s1 * t0) / det
    return fit, deriv


def loo_cv_bandwidth(
    x: FloatArray,
    y: FloatArray,
    grid: FloatArray | None = None,
) -> float:
    """Leave-one-out CV bandwidth for the local-linear smoother."""
    xx, yy = _as_xy(x, y)
    n = xx.size
    bw_grid = (
        np.asarray(grid, dtype=np.float64).ravel()
        if grid is not None
        else np.quantile(np.abs(xx - np.median(xx)), np.linspace(0.08, 0.6, 12))
    )
    bw_grid = np.unique(bw_grid[bw_grid > 0])
    if bw_grid.size < 3:
        raise ValueError("need >=3 distinct positive bandwidths")
    # subsample for speed on large n
    idx = np.arange(n) if n <= 800 else np.linspace(0, n - 1, 800).astype(int)
    errs = []
    for bw in bw_grid:
        e = 0.0
        for i in idx:
            keep = np.ones(n, dtype=bool)
            keep[i] = False
            f, _ = local_linear(xx[keep], yy[keep], xx[i : i + 1], float(bw))
            e += (yy[i] - f[0]) ** 2
        errs.append(e / idx.size)
    return float(bw_grid[int(np.argmin(errs))])


def synth_smooth_curve(
    n: int = 600,
    nonlinear: bool = True,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Smooth-curve DGP: m(x) = sin(2x)·(1+0.3x) when nonlinear —
    local-linear should track curve + derivative; flat control."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.5, 2.5, n)
    m = np.sin(2 * x) * (1 + 0.3 * x) if nonlinear else 0.5 * x
    y = m + rng.normal(0.0, 0.3, n)
    return {"x": x, "y": y, "nonlinear": np.array([float(nonlinear)])}


def bench_kernel_regression(
    seed: int = 20261231 + 216,
) -> dict[str, float]:
    """Kernel-regression self-check: local-linear fit correlation and
    boundary-bias advantage over Nadaraya-Watson. All ``synthetic_*``."""
    d = synth_smooth_curve(nonlinear=True, seed=seed)
    x, y = np.asarray(d["x"]), np.asarray(d["y"])
    bw = loo_cv_bandwidth(x, y)
    xe = np.linspace(-2.0, 2.0, 40)
    f_ll, d_ll = local_linear(x, y, xe, bw)
    f_nw = nadaraya_watson(x, y, xe, bw)
    m_true = np.sin(2 * xe) * (1 + 0.3 * xe)
    dm_true = 2 * np.cos(2 * xe) * (1 + 0.3 * xe) + 0.3 * np.sin(2 * xe)

    rmse_ll = float(np.sqrt(np.mean((f_ll - m_true) ** 2)))
    rmse_nw = float(np.sqrt(np.mean((f_nw - m_true) ** 2)))
    corr_d = float(np.corrcoef(d_ll, dm_true)[0, 1])

    d0 = synth_smooth_curve(nonlinear=False, seed=seed + 1)
    x0, y0 = np.asarray(d0["x"]), np.asarray(d0["y"])
    bw0 = loo_cv_bandwidth(x0, y0)
    f0, d0_ = local_linear(x0, y0, xe, bw0)
    rmse0 = float(np.sqrt(np.mean((f0 - 0.5 * xe) ** 2)))

    f_llb, _ = local_linear(x, y, xe, bw)
    return {
        "synthetic_rmse_ll": rmse_ll,
        "synthetic_rmse_nw": rmse_nw,
        "synthetic_beats_nw": float(rmse_ll < rmse_nw),
        "synthetic_deriv_corr": corr_d,
        "synthetic_cv_bw": bw,
        "synthetic_linear_rmse": rmse0,
        "synthetic_detects": float(rmse_ll < 0.35 and corr_d > 0.5),
        "synthetic_determinism": float(np.allclose(f_ll, f_llb, rtol=0, atol=0)),
    }
