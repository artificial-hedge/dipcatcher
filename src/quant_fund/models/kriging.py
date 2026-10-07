"""Ordinary kriging with fitted variogram — best linear unbiased
spatial prediction (Matheron; Cressie 1993).

Ordinary kriging predicts z(s0) from observations z(s_i) via weights
lambda solving the ordinary-kriging system

    [ Gamma   1 ] [ lambda ]   [ gamma(s0) ]
    [  1^T    0 ] [   mu   ] = [     1     ]

with Gamma_ij = gamma(||s_i - s_j||) the semivariogram. Common models:
spherical, exponential, Gaussian, each (range a, sill c):

    exponential: c (1 - exp(-3h/a)),   Gaussian: c (1 - exp(-3h²/a²)),
    spherical:   c (1.5h/a - 0.5h³/a³) for h < a else c.

The sill/range are fitted to the empirical variogram cloud by
least squares on binned lag means.

Honesty: the bench samples a Gaussian-process surface with known
exponential covariance on a grid, krige a hold-out set, and requires
the RMS prediction error to beat the marginal-sd baseline by a
margin, plus exactness at observed knots. Fail-closed on singular
systems or non-finite input.

References: Matheron (1963) "Principles of geostatistics"; Cressie
(1993) "Statistics for Spatial Data"; Diggle & Ribeiro (2007)
model-based geostatistics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def exponential_variogram(h: FloatArray, rng_a: float, sill: float) -> FloatArray:
    """c (1 - exp(-3h/a))."""
    h = np.asarray(h, dtype=float)
    return np.asarray(sill * (1.0 - np.exp(-3.0 * h / rng_a)), dtype=np.float64)


def empirical_variogram(
    x: FloatArray, y: FloatArray, z: FloatArray, n_bins: int = 15
) -> tuple[FloatArray, FloatArray]:
    """Binned empirical semivariogram: mean (z_i - z_j)^2 / 2 per lag."""
    xx, yy, zz = (np.asarray(v, dtype=float).ravel() for v in (x, y, z))
    n = xx.size
    if n < 10 or not np.isfinite(zz).all():
        raise ValueError("bad input")
    d = np.sqrt((xx[:, None] - xx[None, :]) ** 2 + (yy[:, None] - yy[None, :]) ** 2)
    diff2 = (zz[:, None] - zz[None, :]) ** 2
    iu = np.triu_indices(n, k=1)
    lag = d[iu]
    gam = 0.5 * diff2[iu]
    edges = np.quantile(lag, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.array(
        [
            gam[(lag >= edges[i]) & (lag < edges[i + 1])].mean()
            if np.any((lag >= edges[i]) & (lag < edges[i + 1]))
            else np.nan
            for i in range(len(centers))
        ]
    )
    ok = np.isfinite(means)
    return np.asarray(centers[ok], dtype=np.float64), np.asarray(means[ok], dtype=np.float64)


def fit_variogram(lag: FloatArray, gamma: FloatArray) -> tuple[float, float]:
    """Fit exponential sill+range to binned variogram by grid+refine."""
    h = np.asarray(lag, dtype=float)
    g = np.asarray(gamma, dtype=float)
    if h.size < 5 or not np.isfinite(g).all():
        raise ValueError("bad variogram")
    sill0 = float(max(g.max(), 1e-8))
    best = (sill0, float(np.median(h)))
    best_err = np.inf
    for sill in np.linspace(0.5 * sill0, 1.5 * sill0, 30):
        for a in np.linspace(0.05 * h.max(), 2.0 * h.max(), 60):
            pred = exponential_variogram(h, a, sill)
            err = float(np.sum((pred - g) ** 2))
            if err < best_err:
                best_err = err
                best = (sill, a)
    return best


def ordinary_kriging(
    x: FloatArray,
    y: FloatArray,
    z: FloatArray,
    x_pred: FloatArray,
    y_pred: FloatArray,
    rng_a: float,
    sill: float,
    nugget: float = 0.0,
) -> dict[str, FloatArray | float]:
    """Ordinary kriging at prediction sites. Returns mean and variance."""
    xx, yy, zz = (np.asarray(v, dtype=float).ravel() for v in (x, y, z))
    xp, yp = (np.asarray(v, dtype=float).ravel() for v in (x_pred, y_pred))
    n = xx.size
    if n < 5 or not np.isfinite(zz).all():
        raise ValueError("bad input")
    d = np.sqrt((xx[:, None] - xx[None, :]) ** 2 + (yy[:, None] - yy[None, :]) ** 2)
    gmat = exponential_variogram(d, rng_a, sill) + nugget * np.eye(n)
    a_sys = np.zeros((n + 1, n + 1))
    a_sys[:n, :n] = gmat
    a_sys[:n, n] = 1.0
    a_sys[n, :n] = 1.0
    m = xp.size
    pred = np.empty(m)
    var = np.empty(m)
    for i in range(m):
        d0 = np.sqrt((xx - xp[i]) ** 2 + (yy - yp[i]) ** 2)
        g0 = exponential_variogram(d0, rng_a, sill)
        rhs = np.concatenate([g0, [1.0]])
        sol = np.linalg.solve(a_sys, rhs)
        lam = sol[:n]
        pred[i] = float(lam @ zz)
        var[i] = float(lam @ g0 + sol[n])
    return {
        "mean": np.asarray(pred, dtype=np.float64),
        "variance": np.asarray(np.maximum(var, 0.0), dtype=np.float64),
        "weights_shape": float(n),
    }


def bench_kriging(seed: int = 20261231 + 406) -> dict[str, float]:
    """SYNTHETIC check — kriging beats marginal-sd baseline; exact at knots."""
    rng = np.random.default_rng(seed)
    # n=200: the empirical variogram needs enough pair lags to pin the
    # range before it can be fitted; 90 obs proved too noisy.
    n = 200
    x = rng.uniform(0, 10, n)
    y = rng.uniform(0, 10, n)
    # GP surface with exponential covariance (sill=1, range=3)
    true_range, true_sill = 3.0, 1.0
    d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    cov = true_sill * np.exp(-d / true_range) + 1e-9 * np.eye(n)
    z = rng.multivariate_normal(np.zeros(n), cov)
    # hold out 25% as prediction sites
    hold = rng.choice(n, n // 4, replace=False)
    keep = np.setdiff1d(np.arange(n), hold)
    lag, gam = empirical_variogram(x[keep], y[keep], z[keep])
    sill_f, rng_f = fit_variogram(lag, gam)
    out = ordinary_kriging(x[keep], y[keep], z[keep], x[hold], y[hold], rng_f, sill_f)
    pred = np.asarray(out["mean"])
    rmse_krig = float(np.sqrt(np.mean((pred - z[hold]) ** 2)))
    rmse_base = float(np.std(z[keep]))
    ratio = rmse_krig / rmse_base
    if ratio > 0.8:
        raise ValueError(f"kriging no better than baseline: {ratio}")
    # exactness: krige an observed point -> residual ~0
    out1 = ordinary_kriging(x[keep], y[keep], z[keep], x[keep][:1], y[keep][:1], rng_f, sill_f)
    knot_err = abs(float(np.asarray(out1["mean"])[0]) - z[keep][0])
    if knot_err > 0.05:
        raise ValueError(f"not exact at knot: {knot_err}")
    return {
        "synthetic_kriging_rmse_ratio": ratio,
        "synthetic_kriging_knot_err": float(knot_err),
        "synthetic_kriging_range_fit": float(rng_f),
        "synthetic_score": 1.0,
    }
