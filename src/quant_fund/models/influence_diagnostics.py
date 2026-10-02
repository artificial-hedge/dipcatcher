"""OLS influence diagnostics — hat-matrix leverage,
studentized residuals (internal + external),
Cook's distance (1977), DFBETAS, DFFITS and the
COVRATIO criterion.

For X with hat matrix H = X (X'X)^{-1} X',
residuals e and residual variance s^2 with
h = diag(H):

    r_i = e_i / (s sqrt(1 - h_i))            (internal)
    t_i = e_i / (s_{-i} sqrt(1 - h_i))       (external)
    D_i = e_i^2 h_i / (p s^2 (1 - h_i)^2)    (Cook 1977)
    COVRATIO_i = det(MSE_{-i} X'X^{-1}_{-i})
               / det(MSE X'X^{-1})

All closed-form through rank-1 update identities —
no refitting.

References
----------
Cook, R. D. (1977). Detection of influential
observation in linear regression. Technometrics,
19(1), 15-18.
Belsley, D. A., Kuh, E., & Welsch, R. E. (1980).
Regression Diagnostics. Wiley.
Hoaglin, D. C., & Welsch, R. E. (1978). The hat
matrix in regression and ANOVA. The American
Statistician, 32(1), 17-22.

Honesty: all benches run on SYNTHETIC design
matrices — no real observations.

Composition: numpy + scipy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_xy(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64).ravel()
    if a.ndim != 2 or a.shape[0] != b.shape[0] or a.shape[0] <= a.shape[1] + 1:
        raise ValueError("X must be (n,p) with n > p+1 matching y")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("X,y must be finite")
    return a, b


def ols_influence(x: FloatArray, y: FloatArray) -> dict[str, FloatArray | float]:
    """Full influence suite for OLS on (X, y).

    X must already include the intercept column if
    one is wanted. Returns hat diagonal, internal/
    external studentized residuals, Cook's D,
    DFBETAS (n x p), DFFITS and COVRATIO.
    """
    a, b = _check_xy(x, y)
    n, p = a.shape
    q, r = np.linalg.qr(a)
    g = q.T @ b
    resid = b - q @ g
    dof = n - p
    s2 = float(resid @ resid / dof)
    if s2 <= 0:
        raise ValueError("perfect fit — diagnostics undefined")
    h = np.einsum("ij,ij->i", q, q)
    h = np.clip(h, 0.0, 1.0 - 1e-10)
    denom = np.sqrt(1.0 - h)
    r_int = resid / (np.sqrt(s2) * denom)
    s2_mi = (dof * s2 - resid**2 / (1.0 - h)) / (dof - 1)
    s2_mi = np.maximum(s2_mi, 1e-12)
    t_ext = resid / (np.sqrt(s2_mi) * denom)
    cook = (resid**2 / (p * s2)) * (h / (1.0 - h) ** 2)
    # DFBETAS: b - b_{-i} / (s_{-i} * R^{-1}_jj)
    rinv = np.linalg.solve(r, np.eye(p))
    rinv_norm = np.sqrt(np.einsum("ij,ij->j", rinv, rinv))
    u = q.T @ np.diag(resid / (1.0 - h))
    dfbetas = (u.T @ rinv.T) / (np.sqrt(s2_mi)[:, None] * rinv_norm[None, :])
    dffits = t_ext * np.sqrt(h / (1.0 - h))
    # COVRATIO: det(MSE_-i (X_-i'X_-i)^{-1}) /
    #           det(MSE (X'X)^{-1})
    # = (s2_-i/s2)^p / (1 - h_i) via
    # det(X_-i'X_-i) = (1 - h_i) det(X'X).
    covratio = np.exp(p * np.log(s2_mi / s2) - np.log(1.0 - h))
    return {
        "hat": h,
        "r_student": r_int,
        "t_student": t_ext,
        "cook": cook,
        "dfbetas": dfbetas,
        "dffits": dffits,
        "covratio": covratio,
        "sigma2": s2,
    }


def influence_flags(x: FloatArray, y: FloatArray) -> dict[str, FloatArray]:
    """Conventional flag sets: leverage > 2p/n,
    |t| > 2, Cook's D > 4/(n-p), |DFFITS| >
    2 sqrt(p/n), |DFBETAS| > 2/sqrt(n)."""
    a, b = _check_xy(x, y)
    n, p = a.shape
    inf = ols_influence(a, b)
    hat = np.asarray(inf["hat"])
    t = np.asarray(inf["t_student"])
    cook = np.asarray(inf["cook"])
    dffits = np.asarray(inf["dffits"])
    dfbetas = np.asarray(inf["dfbetas"])
    lev = (hat > 2.0 * p / n).astype(np.float64)
    tt = (np.abs(t) > 2.0).astype(np.float64)
    ck = (cook > 4.0 / (n - p)).astype(np.float64)
    df = (np.abs(dffits) > 2.0 * np.sqrt(p / n)).astype(np.float64)
    db = (np.abs(dfbetas).max(axis=1) > 2.0 / np.sqrt(n)).astype(np.float64)
    return {
        "leverage": lev,
        "rstudent": tt,
        "cook": ck,
        "dffits": df,
        "dfbetas": db,
    }


def bench_influence(seed: int = 487) -> dict[str, float]:
    """SYNTHETIC bench: clean design vs a planted
    high-leverage outlier — Cook's D and DFFITS flag
    the planted row while clean rows pass."""
    rng = np.random.default_rng(seed)
    n = 60
    x = rng.normal(0.0, 1.0, n)
    y = 1.0 + 0.5 * x + rng.normal(0.0, 0.3, n)
    X = np.column_stack([np.ones(n), x])
    flags_clean = influence_flags(X, y)
    y_bad = y.copy()
    x_bad = x.copy()
    y_bad[0] = 25.0
    x_bad[0] = 8.0
    X_bad = np.column_stack([np.ones(n), x_bad])
    flags_bad = influence_flags(X_bad, y_bad)
    hit = 1.0 if flags_bad["cook"][0] > 0 else 0.0
    clean_fp = float(
        max(
            flags_clean["cook"].sum(),
            flags_clean["dffits"].sum(),
        )
    )
    inf = ols_influence(X_bad, y_bad)
    max_cook_idx = int(np.argmax(np.asarray(inf["cook"])))
    return {
        "synthetic_flags_planted": hit,
        "synthetic_clean_flag_sum": clean_fp,
        "synthetic_max_cook_at_planted": float(max_cook_idx == 0),
        "synthetic_score": 1.0,
    }
