"""Robust-regression canon: Huber IRLS, Tukey
biweight S-estimator, least trimmed squares (LTS),
and MM regression (S-start, M-finish).

All estimators take (x, y) design/response and
return coefficient dicts; `bench_robust` checks
recovery of a line under 20% outlier contamination
where OLS visibly fails.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quant_fund._typing import FloatArray

__all__ = [
    "huber_irls",
    "s_estimator",
    "lts",
    "mm_regression",
    "bench_robust",
]


def _ols(x: FloatArray, y: FloatArray) -> FloatArray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _mad(r: FloatArray) -> float:
    return float(np.median(np.abs(r - np.median(r))) / 0.6745)


def huber_irls(
    x: FloatArray,
    y: FloatArray,
    c: float = 1.345,
    it: int = 60,
    tol: float = 1e-8,
) -> dict[str, object]:
    """Huber M-regression via iteratively reweighted
    least squares with MAD scale."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    beta = _ols(x, y)
    scale = _mad(y - x @ beta) + 1e-12
    for _ in range(it):
        r = (y - x @ beta) / scale
        w = np.minimum(1.0, c / np.maximum(np.abs(r), 1e-12))
        xw = x * w[:, None]
        beta_new = np.linalg.lstsq(xw.T @ x + 1e-12 * np.eye(x.shape[1]), xw.T @ y, rcond=None)[0]
        scale_new = _mad(y - x @ beta_new) + 1e-12
        if np.linalg.norm(beta_new - beta) < tol and abs(scale_new - scale) < tol:
            beta, scale = beta_new, scale_new
            break
        beta, scale = beta_new, scale_new
    resid = y - x @ beta
    return {"coef": beta, "scale": scale, "resid": resid}


def _biweight_scale(r: FloatArray, s0: float, c: float, it: int = 30) -> float:
    """M-scale equation for Tukey biweight:
    mean of rho(r/s) = delta."""
    s = max(s0, 1e-9)
    delta = 0.5
    for _ in range(it):
        u = r / s
        a = np.abs(u) < c
        rho = np.where(a, (u**2 / 2) * (1 - u**2 / c**2 + u**4 / (3 * c**4)), c**2 / 6)
        m = float(rho.mean() / (c**2 / 6))
        s = s * np.sqrt(m / delta)
        if not np.isfinite(s):
            break
    return float(s)


def s_estimator(
    x: FloatArray,
    y: FloatArray,
    c: float = 1.547,
    it: int = 80,
    seed: int = 0,
    n_sub: int = 200,
) -> dict[str, object]:
    """Tukey S-regression: minimize biweight M-scale
    via IRLS refinement from random subsample starts."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, k = x.shape
    rng = np.random.default_rng(seed)
    best_beta = _ols(x, y)
    best_s = np.inf
    p_sub = min(k + 1, n)
    for _ in range(n_sub):
        idx = rng.choice(n, p_sub, replace=False)
        beta = _ols(x[idx], y[idx])
        s = _biweight_scale(y - x @ beta, _mad(y - x @ beta), c)
        for _it in range(it):
            r = (y - x @ beta) / (s + 1e-12)
            u = r / c
            w = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
            if w.sum() < k:
                break
            xw = x * w[:, None]
            beta = np.linalg.lstsq(xw.T @ x + 1e-12 * np.eye(k), xw.T @ y, rcond=None)[0]
            s = _biweight_scale(y - x @ beta, s, c)
        if s < best_s:
            best_s, best_beta = s, beta
    return {"coef": best_beta, "scale": best_s}


def lts(
    x: FloatArray,
    y: FloatArray,
    h_frac: float = 0.75,
    it: int = 10,
    seed: int = 0,
    n_starts: int = 100,
) -> dict[str, object]:
    """Least trimmed squares (Rousseeuw): C-steps from
    random p-subset starts, keep best h-subset RSS."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, k = x.shape
    h = max(k + 1, int(n * h_frac))
    rng = np.random.default_rng(seed)
    best_beta = _ols(x, y)
    best_rss = np.inf
    p_sub = min(k + 1, n)
    for _ in range(n_starts):
        idx = rng.choice(n, p_sub, replace=False)
        beta = _ols(x[idx], y[idx])
        prev_rss = np.inf
        for _it in range(it):
            r = np.abs(y - x @ beta)
            idx_h = np.argsort(r)[:h]
            beta = _ols(x[idx_h], y[idx_h])
            rss = float(np.sum(np.sort((y - x @ beta) ** 2)[:h]))
            if rss < best_rss:
                best_rss, best_beta = rss, beta
            if rss >= prev_rss - 1e-12:
                break
            prev_rss = rss
    return {"coef": best_beta, "rss": float(best_rss), "h": h}


def mm_regression(
    x: FloatArray,
    y: FloatArray,
    c_s: float = 1.547,
    c_m: float = 4.685,
    it: int = 60,
    seed: int = 0,
) -> dict[str, object]:
    """MM regression (Yohai 1987): S-estimate for
    scale+start, then biweight M-refinement at
    95%-efficiency tuning."""
    s = s_estimator(x, y, c=c_s, seed=seed)
    beta = np.asarray(s["coef"])
    scale = float(np.asarray(s["scale"])) + 1e-12
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    k = x.shape[1]
    for _ in range(it):
        u = (y - x @ beta) / (scale * c_m)
        w = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
        if w.sum() < k:
            break
        xw = x * w[:, None]
        beta_new = np.linalg.lstsq(xw.T @ x + 1e-12 * np.eye(k), xw.T @ y, rcond=None)[0]
        if np.linalg.norm(beta_new - beta) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return {"coef": beta, "scale": scale}


def bench_robust(seed: int = 537) -> dict[str, float]:
    """SYNTHETIC: y = 1 + 2x with 20% shifted outliers;
    robust methods must hold slope error << OLS."""
    rng = np.random.default_rng(seed)
    n = 100
    xv = rng.uniform(-2, 2, n)
    yv = 1.0 + 2.0 * xv + rng.normal(scale=0.2, size=n)
    n_out = n // 5
    out_idx = rng.choice(n, n_out, replace=False)
    yv[out_idx] += 5.0  # asymmetric contamination
    x = np.column_stack([np.ones(n), xv])
    out: dict[str, float] = {}
    ols_b = _ols(x, yv)
    out["synthetic_ols_slope_err"] = float(abs(ols_b[1] - 2.0))
    ests = {
        "huber": huber_irls(x, yv),
        "s": s_estimator(x, yv, seed=seed),
        "lts": lts(x, yv, seed=seed),
        "mm": mm_regression(x, yv, seed=seed),
    }
    for name, est in ests.items():
        b = np.asarray(est["coef"])
        err = float(abs(b[1] - 2.0))
        out[f"synthetic_{name}_slope_err"] = err
        if err > 0.15:
            raise ValueError(f"{name} slope off: {err:.3f}")
    if out["synthetic_ols_slope_err"] < 3 * out["synthetic_mm_slope_err"]:
        raise ValueError("ols not degraded enough")
    return out
