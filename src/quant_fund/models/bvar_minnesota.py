"""Bayesian VAR with Minnesota (Litterman) prior via dummy observations (SYNTHETIC).

The reduced-form VAR(p) is estimated by augmenting the data with
dummy (Theil mixed-estimation) observations encoding the prior:
own first lag shrinks toward ``mean_own`` (random walk for
persistent series), all other coefficients toward 0, with tightness
controlled by ``lambda_`` and harmonic decay ``lag^decay``.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure shrinkage behavior on generated
VAR panels — never market evidence.

References:
- Litterman (1986). Forecasting with Bayesian vector autoregressions.
  *J. Business & Economic Statistics* 4.
- Doan, Litterman, Sims (1984). Forecasting and conditional
  projection using realistic prior distributions.
  *Econometric Reviews* 3.
- Banbura, Giannone, Reichlin (2010). Large Bayesian VARs.
  *J. Applied Econometrics* 25 (dummy-observation form).
- Karlsson (2013). Forecasting with Bayesian VARs. *Handbook of
  Economic Forecasting* 2B.

Composition: pure numpy — dummy-row augmentation + OLS; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as2(m: FloatArray, name: str) -> FloatArray:
    a = np.asarray(m, dtype=np.float64)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite 2-D matrix required")
    return a


def _design(a: FloatArray, p: int) -> tuple[FloatArray, FloatArray]:
    """Y and X blocks: X row = [1, y_{t-1}..y_{t-p}] (lags concatenated)."""
    t, k = a.shape
    Y = a[p:]
    X = np.column_stack(
        [np.ones(t - p)] + [a[p - i : t - i] for i in range(1, p + 1)],
    )
    return Y, X


def minnesota_dummies(
    k: int,
    p: int,
    lambda_: float = 0.2,
    mean_own: float = 1.0,
    lag_decay: float = 1.0,
    sigma_scales: FloatArray | None = None,
    intercept_tight: float = 1e4,
) -> tuple[FloatArray, FloatArray]:
    """Dummy (Yd, Xd) rows for the Minnesota prior.

    For each variable i and lag l: shrink A_il toward mean_own if
    l==1 else 0, with sd lambda_ * sigma_i / sigma_j / l^decay.
    Intercept is shrunk toward 0 with a very loose sd so it is
    effectively free. sigma_scales are per-variable innovation sd
    estimates (defaults: all 1)."""
    if k < 2 or p < 1:
        raise ValueError("k>=2, p>=1 required")
    if lambda_ <= 0 or not np.isfinite(lambda_):
        raise ValueError("lambda_ must be > 0")
    if lag_decay <= 0:
        raise ValueError("lag_decay must be > 0")
    s = (
        np.ones(k)
        if sigma_scales is None
        else np.asarray(
            sigma_scales,
            dtype=np.float64,
        ).ravel()
    )
    if s.size != k or not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("sigma_scales must be k positive finite values")

    n_coef = 1 + k * p
    rows_y: list[FloatArray] = []
    rows_x: list[FloatArray] = []

    # intercept dummy (loose): y_i = c_i free
    d_y = np.zeros((1, k))
    d_x = np.zeros((1, n_coef))
    d_x[0, 0] = 1.0 / intercept_tight
    rows_y.append(d_y)
    rows_x.append(d_x)

    # lag dummies: one per (variable i, regressor var j, lag l).
    # Each row constrains ONLY equation i: Yd is nonzero solely at
    # column i, and the caller must select rows by their nonzero
    # Yd column — a shared dummy row otherwise penalizes other
    # equations' coefficients toward 0 (contamination).
    for i in range(k):
        for lag in range(1, p + 1):
            for j in range(k):
                sd = lambda_ * s[i] / s[j] / (lag**lag_decay)
                prior_mean = mean_own if (lag == 1 and i == j) else 0.0
                dy = np.zeros(k)
                dy[i] = prior_mean / sd
                dx = np.zeros(n_coef)
                dx[1 + (lag - 1) * k + j] = 1.0 / sd
                rows_y.append(dy.reshape(1, k))
                rows_x.append(dx.reshape(1, n_coef))
    return np.vstack(rows_y), np.vstack(rows_x)


def bvar_estimate(
    y: FloatArray,
    p: int = 1,
    lambda_: float = 0.2,
    mean_own: float = 1.0,
    lag_decay: float = 1.0,
) -> dict[str, FloatArray | int]:
    """BVAR(p) posterior mean = OLS on [Y;Yd] ~ [X;Xd].

    sigma_scales for the prior are AR(1) residual sds per variable."""
    a = _as2(y, "y")
    t, k = a.shape
    if t < p + 3:
        raise ValueError("need t>=p+3")

    # AR(1) residual sd per variable → prior scale ratio
    sig = np.empty(k)
    for i in range(k):
        yi = a[1:, i]
        xi = a[:-1, i]
        b = float(np.cov(xi, yi)[0, 1] / max(np.var(xi), 1e-12))
        sig[i] = max(float(np.std(yi - b * xi)), 1e-6)

    Y, X = _design(a, p)
    Yd, Xd = minnesota_dummies(
        k,
        p,
        lambda_=lambda_,
        mean_own=mean_own,
        lag_decay=lag_decay,
        sigma_scales=sig,
    )
    # Per-equation posterior: only rows whose Yd column matches the
    # equation may enter its regression — plus the loose intercept row.
    n_coef = X.shape[1]
    B = np.zeros((n_coef, k))
    for i in range(k):
        # eq-i dummy block: rows 1+i*k*p .. 1+(i+1)*k*p-1 (row 0 = intercept)
        lo = 1 + i * k * p
        hi = 1 + (i + 1) * k * p
        eq_rows = np.concatenate([[0], np.arange(lo, hi)])
        Xi = np.vstack([X, Xd[eq_rows]])
        yi = np.concatenate([Y[:, i], Yd[eq_rows, i]])
        bi, *_ = np.linalg.lstsq(Xi, yi, rcond=None)
        B[:, i] = bi
    U = Y - X @ B
    return {
        "B": B,
        "sigma_u": (U.T @ U) / max(t - p - 1 - k * p, 1),
        "residuals": U,
        "n_obs": t - p,
        "n_dummies": Yd.shape[0],
        "p": p,
        "k": k,
        "sigma_scales": sig,
    }


def bvar_shrinkage_profile(
    y: FloatArray,
    p: int = 1,
    lambdas: tuple[float, ...] = (0.05, 0.2, 0.5, 1.0),
) -> dict[str, FloatArray]:
    """Own-lag coefficient across the tightness grid — the shrinkage
    curve a user inspects to pick lambda_."""
    a = _as2(y, "y")
    out = np.empty((len(lambdas), a.shape[1]))
    for i, lam in enumerate(lambdas):
        fit = bvar_estimate(a, p=p, lambda_=lam)
        B = np.asarray(fit["B"])
        k = int(fit["k"])
        out[i] = np.diag(B[1 : 1 + k].T)  # own lag-1 coefs
    return {"lambdas": np.asarray(lambdas), "own_lag1": out}


def synth_bvar(
    n: int = 60,
    k: int = 3,
    persistence: float = 0.7,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Short-sample VAR(1) where OLS over-fits noise: BVAR's prior
    should pull own-lag coefs toward persistence and cross coefs to 0."""
    rng = np.random.default_rng(seed)
    A = np.eye(k) * persistence + rng.normal(0.0, 0.05, (k, k))
    np.fill_diagonal(A, persistence)
    y = np.zeros((n, k))
    for t in range(1, n):
        y[t] = A @ y[t - 1] + rng.normal(0.0, 1.0, k)
    return {"y": y, "A": A}


def bench_bvar_minnesota(seed: int = 20261231 + 204) -> dict[str, float]:
    """BVAR self-check: on a short noisy sample, tight prior beats OLS
    on own-lag recovery and kills spurious cross-lags. All ``synthetic_*``."""
    d = synth_bvar(seed=seed)
    y = np.asarray(d["y"])
    A_true = np.asarray(d["A"])
    k = y.shape[1]

    fit = bvar_estimate(y, p=1, lambda_=0.15)
    B = np.asarray(fit["B"])
    A_bvar = B[1 : 1 + k].T  # A_1

    # plain OLS comparison
    Y, X = _design(y, 1)
    Bo, *_ = np.linalg.lstsq(X, Y, rcond=None)
    A_ols = Bo[1 : 1 + k].T

    own_err_bvar = float(np.abs(np.diag(A_bvar) - np.diag(A_true)).mean())
    own_err_ols = float(np.abs(np.diag(A_ols) - np.diag(A_true)).mean())
    cross_bvar = float(np.abs(A_bvar - np.diag(np.diag(A_bvar))).mean())
    cross_ols = float(np.abs(A_ols - np.diag(np.diag(A_ols))).mean())

    prof = bvar_shrinkage_profile(y, p=1)
    lams = np.asarray(prof["lambdas"])
    own = np.asarray(prof["own_lag1"])
    # tighter (smaller lambda) → own-lag closer to 1.0 prior mean
    idx_tight = int(np.argmin(lams))
    idx_loose = int(np.argmax(lams))
    monotone = float(np.all(own[idx_tight] >= own[idx_loose] - 1e-9))

    fit_b = bvar_estimate(y, p=1, lambda_=0.15)

    return {
        "synthetic_own_err_bvar": own_err_bvar,
        "synthetic_own_err_ols": own_err_ols,
        "synthetic_beats_ols_own": float(own_err_bvar < own_err_ols),
        "synthetic_cross_bvar": cross_bvar,
        "synthetic_cross_ols": cross_ols,
        "synthetic_shrinks_cross": float(cross_bvar < cross_ols),
        "synthetic_tightness_monotone": monotone,
        "synthetic_detects": float(own_err_bvar < own_err_ols and cross_bvar < cross_ols),
        "synthetic_determinism": float(
            float(np.asarray(fit["B"])[1, 0]) == float(np.asarray(fit_b["B"])[1, 0])
        ),
    }
