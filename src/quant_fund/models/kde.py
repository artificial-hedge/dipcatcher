"""Kernel density canon: Gaussian KDE with Silverman/normal-reference (SYNTHETIC)
and leave-one-out cross-validated bandwidth selection, plus density
evaluation, CDF via the normal-kernel mixture, and integrated squared
error against a truth. ``bench_kde`` samples a known mixture and gates
LOO-CV bandwidth recovery + L2 error vs the true density.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def silverman_bw(x: FloatArray) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        raise ValueError("need >= 2 points")
    sd = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.quantile(x, [0.75, 0.25]))) / 1.34
    sig = min(sd, iqr) if iqr > 0 else sd
    return float(0.9 * max(sig, 1e-9) * n ** (-0.2))


def kde_eval(x: FloatArray, grid: FloatArray, bw: float) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    if bw <= 0:
        raise ValueError("bw must be positive")
    z = (grid[:, None] - x[None, :]) / bw
    return np.asarray(norm.pdf(z).mean(axis=1) / bw)


def kde_cdf(x: FloatArray, grid: FloatArray, bw: float) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    z = (grid[:, None] - x[None, :]) / bw
    return np.asarray(norm.cdf(z).mean(axis=1))


def _loo_loglik(x: FloatArray, bw: float) -> float:
    n = x.size
    z = (x[:, None] - x[None, :]) / bw
    ker = norm.pdf(z)
    np.fill_diagonal(ker, 0.0)
    dens = ker.sum(axis=1) / ((n - 1) * bw)
    dens = np.maximum(dens, 1e-300)
    return float(np.log(dens).mean())


def loo_cv_bw(x: FloatArray, grid_bw: FloatArray | None = None) -> tuple[float, FloatArray]:
    x = np.asarray(x, dtype=np.float64)
    if grid_bw is None:
        b0 = silverman_bw(x)
        grid_bw = np.linspace(0.3 * b0, 3.0 * b0, 40)
    grid_bw = np.asarray(grid_bw, dtype=np.float64)
    ll = np.array([_loo_loglik(x, b) for b in grid_bw])
    return float(grid_bw[int(np.argmax(ll))]), ll


def bench_kde(seed: int = 20261231) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = 400
    m = rng.uniform(0, 1, n) < 0.6
    x = np.where(m, rng.normal(-1.0, 0.5, n), rng.normal(1.5, 0.4, n))
    grid = np.linspace(-4, 4, 400)
    truth = 0.6 * norm.pdf(grid, -1.0, 0.5) + 0.4 * norm.pdf(grid, 1.5, 0.4)
    b_silv = silverman_bw(x)
    b_cv, _ = loo_cv_bw(x)
    f_silv = kde_eval(x, grid, b_silv)
    f_cv = kde_eval(x, grid, b_cv)
    dx = grid[1] - grid[0]
    cdf = kde_cdf(x, grid, b_cv)
    return {
        "synthetic_bw_silv": b_silv,
        "synthetic_bw_cv": b_cv,
        "synthetic_kde_l2_silv": float(np.sqrt(dx * np.sum((f_silv - truth) ** 2))),
        "synthetic_kde_l2_cv": float(np.sqrt(dx * np.sum((f_cv - truth) ** 2))),
        "synthetic_kde_mass_err": float(abs(dx * f_cv.sum() - 1.0)),
        "synthetic_cdf_tail_err": float(abs(cdf[-1] - 1.0)),
    }
