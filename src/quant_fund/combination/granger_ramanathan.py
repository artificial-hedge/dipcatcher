"""Granger–Ramanathan forecast combination.

Regresses realised values on the member forecasts (with intercept) and uses
the fitted coefficients as combination weights — the classic regression
approach to forecast combination. Includes a constrained variant (negative
weights clipped, renormalised) and a rolling walk-forward scheme so weights
are always estimated out of sample relative to the combined point.

Honesty: combination quality is measured with squared error on the supplied
data; no market claim.

References:
- Granger, C. W. J., Ramanathan, R. (1984). Improved methods of combining
  forecasts — regression-based combination.
- Bates, J. M., Granger, C. W. J. (1969). The combination of forecasts —
  the origin of convex combination weights.
- Clemen, R. T. (1989). Combining forecasts: a review and annotated
  bibliography.

Composition: pure numpy (lstsq); deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _constrain(w: FloatArray) -> FloatArray:
    w = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
    total = float(w.sum())
    if total <= 0:
        return np.full(len(w), 1.0 / len(w))
    return w / total


def granger_ramanathan(
    y: FloatArray,
    preds: FloatArray,
    *,
    ridge: float = 1e-8,
    constrain: bool = True,
) -> dict[str, FloatArray]:
    """OLS combination weights from regressing y on [1, preds].

    ``preds`` is (T, K). Returns ``{"weights": (K,), "intercept": scalar,
    "in_sample_rmse": scalar}``; with ``constrain=True`` weights are clipped
    to the simplex.
    """
    y = np.asarray(y, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    if preds.ndim != 2 or y.ndim != 1 or y.shape[0] != preds.shape[0]:
        raise ValueError("y must be (T,) and preds (T, K) with matching T")
    t, k = preds.shape
    if t <= k + 1:
        raise ValueError("need more observations than members for the regression")
    design = np.column_stack([np.ones(t), preds])
    gram = design.T @ design
    reg = gram + ridge * np.eye(k + 1)
    coef = np.linalg.solve(reg, design.T @ y)
    intercept = float(coef[0])
    w = coef[1:]
    if constrain:
        w = _constrain(w)
    fit = design @ np.concatenate(([intercept], w))
    rmse = float(np.sqrt(np.mean((y - fit) ** 2)))
    return {
        "weights": np.asarray(w, dtype=np.float64),
        "intercept": np.asarray(intercept, dtype=np.float64),
        "in_sample_rmse": np.asarray(rmse, dtype=np.float64),
    }


def rolling_combination(
    y: FloatArray,
    preds: FloatArray,
    *,
    window: int,
    constrain: bool = True,
) -> dict[str, FloatArray]:
    """Walk-forward combination: weights from [t−window, t) score period t.

    Returns the combined forecast for t ≥ window plus the weight path
    (rows sum to 1 when constrained).
    """
    y = np.asarray(y, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    if preds.ndim != 2 or y.ndim != 1 or y.shape[0] != preds.shape[0]:
        raise ValueError("y must be (T,) and preds (T, K) with matching T")
    t_total, k = preds.shape
    if window <= k + 1:
        raise ValueError("window must exceed the number of members")
    n_out = t_total - window
    if n_out <= 0:
        raise ValueError("series too short for the requested window")
    combined = np.empty(n_out, dtype=np.float64)
    weights_path = np.empty((n_out, k), dtype=np.float64)
    for i in range(n_out):
        t = window + i
        fit = granger_ramanathan(y[t - window : t], preds[t - window : t], constrain=constrain)
        w = fit["weights"]
        weights_path[i] = w
        combined[i] = float(preds[t] @ w)
    return {"combined": combined, "weights": weights_path}
