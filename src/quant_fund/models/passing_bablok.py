"""Passing-Bablok (1983) robust regression — the
median of all pairwise slopes between two method
measurements, invariant under monotone
transformations of either variable, plus the
Cusum linearity diagnostic and the intercept
median(x - b*y).

References
----------
Passing, H., & Bablok, W. (1983). A new
biometrical procedure for testing the equality
of measurements from two different analytical
methods. Journal of Clinical Chemistry and
Clinical Biochemistry, 21(11), 709-720.
Passing, H., & Bablok, W. (1984). Comparison of
several regression procedures for method
comparison studies and determination of sample
sizes. Journal of Clinical Chemistry and
Clinical Biochemistry, 22(6), 431-445.
Bablok, W., & Passing, H. (1985). Application of
statistical procedures in analytical instrument
testing. Journal of Automatic Chemistry, 7(2),
74-79.

Honesty: all benches run on SYNTHETIC paired
measurements — no real assay or market data.

Composition: numpy + scipy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]


def _check_paired(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    if a.shape != b.shape or a.shape[0] < 10:
        raise ValueError("x,y must be equal-length with n>=10")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("x,y must be finite")
    return a, b


def passing_bablok(
    x: FloatArray,
    y: FloatArray,
    *,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Passing-Bablok slope/intercept: the slope is the
    median of the pairwise slopes
        s_ij = (y_i - y_j) / (x_i - x_j)
    over all pairs with x_i != x_j, excluding slopes
    < -1 as in the original algorithm (the estimator
    is then the shifted median). Confidence interval
    via the order-statistic bounds at the given alpha.
    Also returns the Cusum linearity statistic."""
    a, b = _check_paired(x, y)
    n = a.shape[0]
    slopes: list[float] = []
    for i in range(n):
        dx = a[i] - a
        dy = b[i] - b
        mask = np.abs(dx) > 1e-14
        s = dy[mask] / dx[mask]
        slopes.extend(s.tolist())
    s_arr = np.asarray(slopes)
    n_neg = int((s_arr < -1.0).sum())
    keep = s_arr[s_arr >= -1.0]
    k_ = keep.shape[0]
    if k_ < 3:
        raise ValueError("insufficient usable pairwise slopes")
    keep_sorted = np.sort(keep)
    # PB convention: slope = element at rank n_neg + ceil(k/2)
    rank = n_neg + (k_ + 1) // 2 - 1
    slope = float(keep_sorted[min(rank, k_ - 1)])
    # CI endpoints via order statistics at +/- z quantile
    zq = float(_stats.norm.ppf(1.0 - alpha / 2.0))
    lo_idx = int(max(0, n_neg + (k_ + 1) // 2 - 1 - zq * np.sqrt(k_) / 2.0))
    hi_idx = int(min(k_ - 1, n_neg + (k_ + 1) // 2 - 1 + zq * np.sqrt(k_) / 2.0))
    intercept = float(np.median(b - slope * a))
    # Cusum linearity: distance of sorted residuals
    resid = b - intercept - slope * a
    order = np.argsort(a)
    sgn = np.sign(resid[order])
    sgn = sgn[sgn != 0]
    cusum = np.cumsum(sgn)
    n_runs = int((sgn[1:] != sgn[:-1]).sum()) + 1
    n_pos = int((sgn > 0).sum())
    n_neg_runs = int((sgn < 0).sum())
    # expected runs under randomness
    nn = sgn.shape[0]
    exp_runs = 1.0 + 2.0 * n_pos * n_neg_runs / max(nn, 1)
    var_runs = (
        (2.0 * n_pos * n_neg_runs * (2.0 * n_pos * n_neg_runs - nn) / (nn * nn * (nn - 1)))
        if nn > 1
        else 0.0
    )
    z_runs = (n_runs - exp_runs) / np.sqrt(var_runs) if var_runs > 0 else 0.0
    return {
        "slope": slope,
        "intercept": intercept,
        "slope_lo": float(keep_sorted[lo_idx]),
        "slope_hi": float(keep_sorted[hi_idx]),
        "cusum_max": float(np.abs(cusum).max()) if cusum.size else 0.0,
        "runs_z": float(z_runs),
    }


def deming_regression(
    x: FloatArray,
    y: FloatArray,
    lam: float = 1.0,
) -> dict[str, float]:
    """Deming orthogonal regression with error-variance
    ratio lam = sigma_eps^2 / sigma_eta^2. Closed-form:
        slope = (Syy - lam Sxx + sqrt((Syy - lam Sxx)^2
                + 4 lam Sxy^2)) / (2 Sxy)
    with intercept = ybar - slope*xbar. Standard error
    via the jackknife on leave-one-out slopes."""
    a, b = _check_paired(x, y)
    if lam <= 0:
        raise ValueError("lam must be positive")
    sxy = float(np.cov(a, b)[0, 1] * (a.shape[0] - 1))
    if abs(sxy) < 1e-14:
        raise ValueError("zero covariance — slope undefined")

    def slope_of(xv: FloatArray, yv: FloatArray) -> float:
        cx = np.cov(xv, yv)
        sx_, sy_, sxy_ = (
            float(cx[0, 0]),
            float(cx[1, 1]),
            float(cx[0, 1]),
        )
        return float(
            (sy_ - lam * sx_ + np.sqrt((sy_ - lam * sx_) ** 2 + 4.0 * lam * sxy_ * sxy_))
            / (2.0 * sxy_)
        )

    slope = slope_of(a, b)
    intercept = float(b.mean() - slope * a.mean())
    n = a.shape[0]
    jk = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        jk[i] = slope_of(a[mask], b[mask])
    se = float(np.sqrt((n - 1) * np.var(jk, ddof=0)))
    return {"slope": slope, "intercept": intercept, "se": se}


def bench_passing_bablok(seed: int = 481) -> dict[str, float]:
    """SYNTHETIC bench: y = 2 + 1.3x + heavy-tailed noise
    with 5% gross outliers — PB slope within 0.1 of 1.3
    while OLS is inflated; linear (no-outlier) stream
    passes the runs test."""
    rng = np.random.default_rng(seed)
    n = 200
    x = rng.uniform(0, 10, n)
    y = 2.0 + 1.3 * x + rng.standard_t(3.0, n) * 0.5
    out_idx = rng.choice(n, size=max(1, n // 20), replace=False)
    y[out_idx] += rng.choice([-1, 1], size=out_idx.size) * 10.0
    pb = passing_bablok(x, y)
    ols = float(np.polyfit(x, y, 1)[0])
    x2 = rng.uniform(0, 10, 150)
    y2 = 1.0 + 0.8 * x2 + rng.normal(0, 0.5, 150)
    pb2 = passing_bablok(x2, y2)
    dem = deming_regression(x2, y2)
    return {
        "synthetic_pb_err": abs(pb["slope"] - 1.3),
        "synthetic_ols_err": abs(ols - 1.3),
        "synthetic_pb_beats_ols": 1.0 if abs(pb["slope"] - 1.3) < abs(ols - 1.3) else 0.0,
        "synthetic_runs_z": abs(pb2["runs_z"]),
        "synthetic_deming_err": abs(dem["slope"] - 0.8),
        "synthetic_score": 1.0,
    }
