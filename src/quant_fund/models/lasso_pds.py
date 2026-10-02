"""Belloni-Chernozhukov-Hansen post-double-selection LASSO.

References
----------
- Belloni, A., Chernozhukov, V. & Hansen, C. (2014).
  "Inference on Treatment Effects After Selection Among
  High-Dimensional Controls." *Review of Economic Studies*
  81(2), 608-650.
- Belloni, A., Chen, D., Chernozhukov, V. & Hansen, C.
  (2012). "Sparse Models and Methods for Optimal
  Instruments with an Application to Eminent Domain."
  *Econometrica* 80(6), 2369-2429.
- Tibshirani, R. (1996). "Regression Shrinkage and
  Selection via the Lasso." *JRSS-B* 58(1), 267-288.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
Partial-linear ``y = d*D + g(X) + e``: naive post-selection
on the outcome equation alone misses controls correlated
with D and biases the t-test. Double selection LASSOs BOTH
``y ~ X`` and ``D ~ X`` and refits on the union support —
the coefficient on D is then regular at sqrt(n) under
approximate sparsity. The penalty is the BCCH data-driven
level ``lam = 2 c sigma_hat sqrt(n) Phi^{-1}(1 - alpha/2p)``
(c=1.1, alpha=0.05) implemented with a cyclic
coordinate-descent soft-thresholding solver on standardized
columns. The bench plants a sparse g on p=100 controls with
n=150: post-PDS interval must cover the planted effect, the
naive full-OLS baseline (overfit SE inflation) may not, and
the union support must retain >= 4/5 true controls.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, t

FloatArray = NDArray[np.float64]


def _soft(z: float, thr: float) -> float:
    return float(np.sign(z) * max(abs(z) - thr, 0.0))


def _lasso_cd(
    y: FloatArray,
    x: FloatArray,
    lam: float,
    iters: int = 500,
) -> FloatArray:
    """Cyclic coordinate-descent LASSO.

    Objective ``(1/2n)||y - X b||^2 + lam * ||b||_1`` on
    columns standardized to unit variance (``x_j' x_j / n = 1``);
    returns coefficients in original x units.
    """
    n, p = x.shape
    sd = np.maximum(x.std(axis=0), 1e-12)
    xs = x / sd
    ys = y - y.mean()
    beta = np.zeros(p)
    r = ys.copy()
    for _ in range(iters):
        delta = 0.0
        for j in range(p):
            col = xs[:, j]
            rho = float(col @ (r + col * beta[j]) / n)
            bj = _soft(rho, lam)
            delta = max(delta, abs(bj - beta[j]))
            r += col * (beta[j] - bj)
            beta[j] = bj
        if delta < 1e-10:
            break
    return beta / sd


def _bcch_lam(y: FloatArray, x: FloatArray, alpha: float = 0.05) -> float:
    n, p = x.shape
    # pilot residual scale: OLS on the top-|corr| controls is
    # the BCCH plug-in for the noise level (not std(y)).
    k0 = min(8, p)
    top = np.argsort(-np.abs((x - x.mean(0)).T @ (y - y.mean())))[:k0]
    xm0 = np.column_stack([np.ones(n), x[:, top]])
    cf, *_ = np.linalg.lstsq(xm0, y, rcond=None)
    sigma_hat = max(float(np.std(y - xm0 @ cf)), 1e-3) * 1.1
    # BCCH's sum-form penalty lam_bcch = 2 c sigma sqrt(n) z
    # translated to the (1/2n)-normalized objective = lam_bcch / n.
    return float(2 * 1.1 * sigma_hat * norm.ppf(1 - alpha / (2 * p)) / np.sqrt(n))


def post_double_selection(
    y: FloatArray,
    d: FloatArray,
    x: FloatArray,
) -> dict[str, float]:
    """BCCH post-double-selection inference on ``d``."""
    yy = np.asarray(y, dtype=np.float64)
    dd = np.asarray(d, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    if yy.ndim != 1 or dd.ndim != 1 or xx.ndim != 2 or yy.size != dd.size or xx.shape[0] != yy.size:
        raise ValueError("bad inputs")
    if not (np.all(np.isfinite(yy)) and np.all(np.isfinite(dd)) and np.all(np.isfinite(xx))):
        raise ValueError("nonfinite")
    if float(np.std(dd)) < 1e-12:
        raise ValueError("degenerate treatment")
    n = yy.size
    lam_y = _bcch_lam(yy, xx)
    lam_d = _bcch_lam(dd, xx)
    by = _lasso_cd(yy, xx, lam_y)
    bd = _lasso_cd(dd, xx, lam_d)
    sel_y = np.abs(by) > 1e-8
    sel_d = np.abs(bd) > 1e-8
    supp = sel_y | sel_d
    xs = xx[:, supp] if supp.any() else np.zeros((n, 0))
    xm = np.column_stack([np.ones(n), dd, xs])
    coef, *_ = np.linalg.lstsq(xm, yy, rcond=None)
    resid = yy - xm @ coef
    dof = max(n - xm.shape[1], 5)
    s2 = float(resid @ resid / dof)
    cov = s2 * np.linalg.pinv(xm.T @ xm)
    se = float(np.sqrt(max(cov[1, 1], 1e-30)))
    crit = float(t.ppf(0.975, dof))
    return {
        "d_hat": float(coef[1]),
        "se": se,
        "ci_lo": float(coef[1] - crit * se),
        "ci_hi": float(coef[1] + crit * se),
        "n_sel": float(supp.sum()),
        "t_stat": float(coef[1] / se),
    }


def synth_pds(
    seed: int = 20261231 + 335,
    n: int = 150,
    p: int = 100,
    s: int = 5,
    d_true: float = 1.0,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC sparse partial-linear: returns (y, d, x, true_supp)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    supp = np.arange(s)
    g = x[:, :s] @ rng.uniform(1.5, 2.5, s)
    h = x[:, :s] @ rng.uniform(0.8, 1.2, s)
    dd = h + rng.standard_normal(n)
    y = d_true * dd + g + rng.standard_normal(n)
    return y, dd, x, np.asarray(supp, dtype=np.float64)


def bench_lasso_pds(seed: int = 20261231 + 335) -> dict[str, float]:
    """Wave-57 self-check: PDS interval covers the planted effect."""
    y, d, x, supp = synth_pds(seed=seed)
    r = post_double_selection(y, d, x)
    covers = float(r["ci_lo"] <= 1.0 <= r["ci_hi"])
    bias = abs(r["d_hat"] - 1.0)
    # support check via the two lasso paths
    by = _lasso_cd(y, x, _bcch_lam(y, x))
    bd = _lasso_cd(d, x, _bcch_lam(d, x))
    found = int(((np.abs(by) > 1e-8) | (np.abs(bd) > 1e-8))[:5].sum())
    ok = covers == 1.0 and bias < 0.35 and found >= 4
    return {
        "d_hat": r["d_hat"],
        "ci_lo": r["ci_lo"],
        "ci_hi": r["ci_hi"],
        "n_sel": r["n_sel"],
        "true_found": float(found),
        "score": float(ok),
    }
