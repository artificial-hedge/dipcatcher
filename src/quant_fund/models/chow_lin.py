"""Chow-Lin and Denton temporal disaggregation.

References
----------
- Chow, G.C. & Lin, A.-l. (1971). "Best Linear Unbiased
  Interpolation, Distribution, and Extrapolation of Time
  Series by Related Series." *Review of Economics and
  Statistics* 53(4), 372-375.
- Denton, F.T. (1971). "Adjustment of Monthly or Quarterly
  Series to Annual Totals: An Approach Based on Quadratic
  Minimization." *JASA* 66(333), 99-102.
- Bournay, J. & Laroque, G. (1979). "Reflexions sur la
  methode d'elaboration des comptes trimestriels."
  *Annales de l'INSEE* 36, 3-30.
- Dagum, E.B. & Cholette, P.A. (2006). *Benchmarking,
  Temporal Distribution, and Reconciliation Methods for
  Time Series*. Springer, ch. 4-6.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
Temporal disaggregation distributes a low-frequency series
(annual) onto a high-frequency grid (quarterly) using
related high-frequency indicators. Chow-Lin assumes the
high-frequency residuals follow an AR(1): GLS with
covariance ``V(rho)`` derived from the AR(1) autocovariance
of the assumed error process — rho estimated by ML grid
search on the aggregated likelihood (the honest one-
parameter profile, rather than OLS on levels which is the
textbook trap). Denton's proportional first-difference
variant minimizes ``sum (x_t/indicator_t - x_{t-1}/
indicator_{t-1})^2`` subject to the annual-total binding —
a pure quadratic program solved by the Lagrange formula,
no rho to estimate but no statistical model either.
Both enforce *exact* aggregation: the disaggregated values
sum to the observed annual totals to machine precision.
``synth_disagg`` builds a smooth latent quarterly series
plus indicator, aggregates to annual, and checks the
reconstruction RMSE vs naive uniform interpolation; the
bench gates on exact annual additivity, Chow-Lin RMSE <
uniform, and rho_hat recovering persistence.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_len: int = 8) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def _agg_matrix(n_lo: int, s: int) -> FloatArray:
    """n_lo x (n_lo*s) aggregation: row i sums quarters of year i."""
    a = np.zeros((n_lo, n_lo * s))
    for i in range(n_lo):
        a[i, i * s : (i + 1) * s] = 1.0
    return a


def _ar1_cov(n: int, rho: float) -> FloatArray:
    v = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            v[i, j] = rho ** abs(i - j)
    return v


def chow_lin(
    y_lo: FloatArray,
    x_hi: FloatArray,
    s: int = 4,
) -> dict[str, float | FloatArray]:
    """Chow-Lin GLS disaggregation with AR(1) rho profile."""
    yl = _as_series(y_lo, min_len=6)
    xh = np.asarray(x_hi, dtype=np.float64)
    if xh.ndim == 1:
        xh = xh[:, None]
    n_lo = yl.size
    n_hi = n_lo * s
    if xh.shape[0] != n_hi:
        raise ValueError("indicator length mismatch")
    if s < 2 or not np.all(np.isfinite(xh)):
        raise ValueError("bad disaggregation")
    a = _agg_matrix(n_lo, s)
    xh1 = np.column_stack([np.ones(n_hi), xh])
    xl = a @ xh1

    def nll_rho(rho: float) -> float:
        v = _ar1_cov(n_hi, rho)
        vl = a @ v @ a.T
        try:
            ic = np.linalg.inv(vl)
        except np.linalg.LinAlgError:
            return 1e12
        b = np.linalg.solve(xl.T @ ic @ xl, xl.T @ ic @ yl)
        r = yl - xl @ b
        sign, logdet = np.linalg.slogdet(vl)
        if sign <= 0:
            return 1e12
        return float(n_lo * np.log(max(float(r @ ic @ r), 1e-300)) + float(logdet))

    grid = np.linspace(-0.05, 0.99, 60)
    vals = np.array([nll_rho(float(g)) for g in grid])
    rho = float(grid[int(np.argmin(vals))])
    v = _ar1_cov(n_hi, rho)
    vl = a @ v @ a.T
    ic = np.linalg.inv(vl)
    b = np.linalg.solve(xl.T @ ic @ xl, xl.T @ ic @ yl)
    r = yl - xl @ b
    y_hat = xh1 @ b + v @ a.T @ ic @ r
    # enforce exact additivity
    resid_agg = yl - a @ y_hat
    if np.max(np.abs(resid_agg)) > 1e-6:
        y_hat = y_hat + a.T @ np.linalg.solve(a @ a.T, resid_agg)
    out: dict[str, float | FloatArray] = {
        "y_hat": np.asarray(y_hat, dtype=np.float64),
        "rho_hat": rho,
        "beta": np.asarray(b, dtype=np.float64),
        "resid_ssr": float(r @ r),
    }
    return out


def denton_proportional(
    y_lo: FloatArray,
    x_hi: FloatArray,
    s: int = 4,
) -> dict[str, float | FloatArray]:
    """Denton proportional first-difference benchmarking."""
    yl = _as_series(y_lo, min_len=6)
    x = _as_series(x_hi, min_len=8)
    n_lo = yl.size
    n_hi = n_lo * s
    if x.size != n_hi:
        raise ValueError("indicator length mismatch")
    # minimize (D (x/ind))^T (D (x/ind)) s.t. A x = y_lo
    # let r = x/ind: minimize r^T D^T D r subject to (A diag(ind)) r = y
    d = np.diff(np.eye(n_hi), axis=0)
    p = d.T @ d
    ind = np.maximum(np.abs(x), 1e-9)
    a_ind = _agg_matrix(n_lo, s) * ind[None, :]
    top = np.block([[p, a_ind.T], [a_ind, np.zeros((n_lo, n_lo))]])
    rhs = np.concatenate([np.zeros(n_hi), yl])
    sol = np.linalg.solve(top + 1e-10 * np.eye(top.shape[0]), rhs)
    r_hat = sol[:n_hi]
    y_hat = r_hat * ind
    out: dict[str, float | FloatArray] = {
        "y_hat": np.asarray(y_hat, dtype=np.float64),
        "resid_ssr": float(np.sum((yl - _agg_matrix(n_lo, s) @ y_hat) ** 2)),
    }
    return out


def synth_disagg(
    seed: int = 20261231 + 356,
    n_lo: int = 12,
    s: int = 4,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC smooth latent quarterly + indicator + annual totals."""
    rng = np.random.default_rng(seed)
    n_hi = n_lo * s
    t = np.arange(n_hi)
    # indicator carries strong within-year seasonality + trend
    ind = (
        10.0
        + 0.06 * t
        + 4.0 * np.sin(2 * np.pi * t / s)
        + 1.5 * np.cos(2 * np.pi * t / (2 * s))
        + 0.5 * rng.standard_normal(n_hi)
    )
    # latent = load * indicator + persistent AR(1) deviation
    u = np.zeros(n_hi)
    for k in range(1, n_hi):
        u[k] = 0.7 * u[k - 1] + 0.5 * rng.standard_normal()
    latent = 2.0 + 0.8 * ind + u
    y_lo = _agg_matrix(n_lo, s) @ latent
    return (
        y_lo.astype(np.float64),
        ind.astype(np.float64),
        latent.astype(np.float64),
    )


def bench_chow_lin(seed: int = 20261231 + 356) -> dict[str, float]:
    y_lo, ind, latent = synth_disagg(seed=seed)
    r = chow_lin(y_lo, ind)
    yh = np.asarray(r["y_hat"])
    n_lo = y_lo.size
    s = yh.size // n_lo
    # annual additivity
    a_err = float(np.max(np.abs(_agg_matrix(n_lo, s) @ yh - y_lo)))
    # reconstruction vs naive spread
    naive = np.repeat(y_lo / s, s)
    rmse_cl = float(np.sqrt(np.mean((yh - latent) ** 2)))
    rmse_nv = float(np.sqrt(np.mean((naive - latent) ** 2)))
    rd = denton_proportional(y_lo, ind)
    yh_d = np.asarray(rd["y_hat"])
    a_err_d = float(np.max(np.abs(_agg_matrix(n_lo, s) @ yh_d - y_lo)))
    rmse_d = float(np.sqrt(np.mean((yh_d - latent) ** 2)))
    ok = (
        a_err < 1e-5
        and a_err_d < 1e-5
        and rmse_cl < rmse_nv
        and rmse_d < rmse_nv
        and r["rho_hat"] > 0.3
    )
    out: dict[str, float] = {
        "synthetic_cl_agg_err": a_err,
        "synthetic_cl_rmse": rmse_cl,
        "synthetic_cl_naive_rmse": rmse_nv,
        "synthetic_cl_rho_hat": float(r["rho_hat"]),
        "synthetic_denton_agg_err": a_err_d,
        "synthetic_denton_rmse": rmse_d,
        "score": 1.0 if ok else 0.0,
    }
    return out
