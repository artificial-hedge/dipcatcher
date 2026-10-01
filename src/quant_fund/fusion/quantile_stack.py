"""Leakage-safe out-of-fold quantile fusion (pinball-weighted stacking).

Distributional heads emit quantile grids; ``fusion.engine`` stacks point
signals under squared loss. This module is the quantile counterpart: per
quantile level τ it fits a non-negative member blend minimizing the
in-fold pinball loss (iteratively reweighted ridge on the check-function
weights), produces out-of-fold stacked quantiles, and repairs crossings
by the Chernozhukov–Fernández-Val rearrangement (per-row sort).

The fold contract matches ``cross_fitted_ridge_stack``: chronological
train/test index pairs, no train/test overlap, every test index covered
exactly once; violations fail closed. Research machinery only — a better
OOF pinball is evidence, not promotion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_IRLS_ITERS = 40
_EPS = 1e-6


@dataclass(frozen=True)
class QuantileStackResult:
    """OOF stacked quantile grid plus per-member pinball attribution."""

    weights: NDArray[np.float64]  # (n_taus, n_members) non-negative
    intercepts: NDArray[np.float64]  # (n_taus,)
    oof_quantiles: NDArray[np.float64]  # (n, n_taus), NaN on warm-up rows
    fold_ids: NDArray[np.int64]
    taus: NDArray[np.float64]
    alpha: float
    oof_pinball: NDArray[np.float64]  # (n_taus,) stacked mean pinball
    member_pinball: NDArray[np.float64]  # (n_taus, n_members)


def _pinball(tau: float, resid: NDArray[np.float64]) -> NDArray[np.float64]:
    return resid * (tau - (resid < 0.0))


def _pinball_ridge(
    x: NDArray[np.float64], y: NDArray[np.float64], tau: float, alpha: float
) -> tuple[NDArray[np.float64], float]:
    """Non-negative pinball-regression blend via IRLS on the check loss.

    Design matrix is [1, X] with an unpenalized intercept; member weights
    are ridge-penalized and projected to the non-negative orthant each
    iteration (approximate but stable; the projection bias is small for
    ensemble members that are already reasonable quantile predictors)."""
    n, m = x.shape
    a = np.column_stack([np.ones(n), x])
    pen = np.diag([0.0] + [alpha * n] * m)
    beta = np.concatenate([[float(np.quantile(y, tau))], np.full(m, 1.0 / m)])
    for _ in range(_IRLS_ITERS):
        resid = y - a @ beta
        iw = np.abs(tau - (resid < 0.0)) + _EPS
        sw = np.sqrt(iw)
        aw = a * sw[:, None]
        gram = aw.T @ aw + pen
        try:
            beta_new = np.linalg.solve(gram, aw.T @ (y * sw))
        except np.linalg.LinAlgError as exc:
            raise ValueError("pinball stacker fit is singular") from exc
        beta_new[1:] = np.clip(beta_new[1:], 0.0, None)
        if not np.isfinite(beta_new).all():
            raise ValueError("pinball stacker fit produced non-finite parameters")
        if np.allclose(beta_new, beta, atol=1e-10):
            beta = beta_new
            break
        beta = beta_new
    return beta[1:], float(beta[0])


def cross_fitted_quantile_stack(
    member_quantiles: NDArray[np.float64],
    target: NDArray[np.float64],
    taus: NDArray[np.float64],
    folds: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
    *,
    alpha: float = 1.0,
) -> QuantileStackResult:
    """OOF pinball stacker over member quantile predictions.

    ``member_quantiles`` is (n, n_members, n_taus); ``target`` is (n,);
    ``taus`` strictly increasing in (0, 1). Same fold contract as
    ``cross_fitted_ridge_stack``. Warm-up rows keep NaN OOF quantiles.
    """
    q = np.asarray(member_quantiles, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    t = np.asarray(taus, dtype=float).reshape(-1)
    if q.ndim != 3 or q.shape[0] != y.size or q.shape[2] != t.size or not folds:
        raise ValueError(
            "quantile stacker requires (n, members, taus) predictions, "
            "a matching target, taus, and folds"
        )
    if q.shape[1] < 1:
        raise ValueError("quantile stacker requires at least one member")
    if t.size < 1 or np.any(t <= 0.0) or np.any(t >= 1.0) or np.any(np.diff(t) <= 0.0):
        raise ValueError("taus must be strictly increasing inside (0, 1)")
    if not np.isfinite(q).all() or not np.isfinite(y).all():
        raise ValueError("quantile stacker inputs must be finite")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("quantile stacker alpha must be finite and positive")

    n, m, k = q.shape
    oof = np.full((n, k), np.nan, dtype=float)
    fold_ids = np.full(n, -1, dtype=np.int64)
    for fold_id, (train_raw, test_raw) in enumerate(folds):
        train = np.asarray(train_raw, dtype=np.int64).reshape(-1)
        test = np.asarray(test_raw, dtype=np.int64).reshape(-1)
        if train.size < 2 or test.size == 0:
            raise ValueError("each stacker fold needs training and test observations")
        if np.any(train < 0) or np.any(test < 0) or np.any(train >= n) or np.any(test >= n):
            raise ValueError("stacker fold index is out of bounds")
        if np.intersect1d(train, test).size:
            raise ValueError("stacker train/test fold overlap would leak targets")
        if np.unique(test).size != test.size or np.any(fold_ids[test] != -1):
            raise ValueError("stacker test observations must be covered exactly once")
        for j, tau in enumerate(t):
            w, b = _pinball_ridge(q[train, :, j], y[train], float(tau), alpha)
            oof[test, j] = q[test, :, j] @ w + b
        fold_ids[test] = fold_id
    eligible = fold_ids >= 0
    if np.count_nonzero(eligible) < 2:
        raise ValueError("stacker folds must provide at least two scored observations")

    # Rearrangement repair: sort each OOF row across taus so the stacked
    # grid is monotone even when per-level fits disagree on ordering.
    oof[eligible] = np.sort(oof[eligible], axis=1)

    weights = np.full((k, m), np.nan)
    intercepts = np.full(k, np.nan)
    for j, tau in enumerate(t):
        w, b = _pinball_ridge(q[eligible, :, j], y[eligible], float(tau), alpha)
        weights[j], intercepts[j] = w, b

    oof_pinball = np.array(
        [
            float(np.mean(_pinball(float(tau), y[eligible] - oof[eligible, j])))
            for j, tau in enumerate(t)
        ]
    )
    member_pinball = np.array(
        [
            [
                float(np.mean(_pinball(float(tau), y[eligible] - q[eligible, mi, j])))
                for mi in range(m)
            ]
            for j, tau in enumerate(t)
        ]
    )
    return QuantileStackResult(
        weights=weights,
        intercepts=intercepts,
        oof_quantiles=oof,
        fold_ids=fold_ids,
        taus=t,
        alpha=float(alpha),
        oof_pinball=oof_pinball,
        member_pinball=member_pinball,
    )
