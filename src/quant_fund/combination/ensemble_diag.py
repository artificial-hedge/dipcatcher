"""Ensemble diagnostics: member error correlation and encompassing.

Before combining, it helps to know whether members carry complementary
information:

- ``error_correlation`` — pairwise correlation matrix of member forecast
  errors; low/negative correlations mark diversity that combination can
  exploit;
- ``encompassing_test`` — the Harvey–Liu (2023) style encompassing check:
  regress the target on the combined forecast and each member's deviation
  from it; significant deviation coefficients mean the member adds
  information beyond the combination;
- ``diversity_report`` — mean pairwise error correlation plus the
  combination-puzzle summary (best member vs equal-weight MSE).

Honesty: correlations and encompassing regressions are descriptive on the
supplied evaluation window.

References:
- Harvey, D. I., Liu, Z. (2023). Encompassing tests for the forecast
  combination puzzle — the regression form used here.
- Clemen, R. T. (1989). Combining forecasts — diversity and the puzzle.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def error_correlation(member_preds: FloatArray, y: FloatArray) -> FloatArray:
    """Correlation matrix of member errors e_k = y − f_k.

    ``member_preds`` is (T, M); ``y`` is (T,). Returns the (M, M) matrix.
    """
    f = np.asarray(member_preds, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if f.ndim != 2 or y_arr.shape != (f.shape[0],):
        raise ValueError("member_preds must be (T, M) and y must be (T,)")
    e = y_arr[:, None] - f
    e = e - e.mean(axis=0, keepdims=True)
    cov = e.T @ e / len(e)
    d = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(d, d)
    return np.asarray(np.clip(corr, -1.0, 1.0), dtype=np.float64)


def encompassing_test(
    member_preds: FloatArray,
    weights: FloatArray,
    y: FloatArray,
) -> dict[str, FloatArray | float]:
    """Encompassing regression of y on the combination and member deviations.

    Model: y = α + β·f_comb + Σ_k γ_k (f_k − f_comb) + ε. Under the null
    that the combination encompasses all members, every γ_k = 0. Returns
    the γ estimates, their t-statistics, and the regression R².
    """
    f = np.asarray(member_preds, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if f.ndim != 2 or y_arr.shape != (f.shape[0],):
        raise ValueError("member_preds must be (T, M) and y must be (T,)")
    m = f.shape[1]
    if w.shape != (m,):
        raise ValueError("weights must be (M,)")
    comb = f @ w
    dev = f - comb[:, None]
    design = np.column_stack([np.ones(len(y_arr)), comb, dev])
    coef, _, _, _ = np.linalg.lstsq(design, y_arr, rcond=None)
    resid = y_arr - design @ coef
    dof = len(y_arr) - design.shape[1]
    sigma2 = float(np.sum(resid * resid)) / max(dof, 1)
    cov = sigma2 * np.linalg.inv(design.T @ design + 1e-10 * np.eye(design.shape[1]))
    se = np.sqrt(np.diag(cov))
    gamma = coef[2:]
    t_gamma = gamma / se[2:]
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    return {
        "gamma": np.asarray(gamma, dtype=np.float64),
        "t_gamma": np.asarray(t_gamma, dtype=np.float64),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
    }


def diversity_report(member_preds: FloatArray, y: FloatArray) -> dict[str, float]:
    """Mean pairwise error correlation plus equal-weight vs best member."""
    f = np.asarray(member_preds, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    corr = error_correlation(f, y_arr)
    iu = np.triu_indices_from(corr, k=1)
    mean_corr = float(np.mean(corr[iu])) if len(iu[0]) else 0.0
    mse = np.mean((f - y_arr[:, None]) ** 2, axis=0)
    equal_mse = float(np.mean((f.mean(axis=1) - y_arr) ** 2))
    best_mse = float(np.min(mse))
    return {
        "mean_error_corr": mean_corr,
        "equal_weight_mse": equal_mse,
        "best_member_mse": best_mse,
        "equal_beats_best": 1.0 if equal_mse < best_mse else 0.0,
        "n_members": float(f.shape[1]),
    }
