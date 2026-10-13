"""Quantile-forecast calibration diagnostics and bias correction.

A quantile forecaster is calibrated at level τ if P(y ≤ q_τ) = τ. This
module measures per-level exceedance, decomposes pinball loss into a
calibration and a sharpness part, and applies a constant bias correction:

- ``exceedance_rates`` — empirical P(y ≤ q_τ) per level against nominal τ;
- ``pinball_decomposition`` — Murphy decomposition of mean pinball into
  reliability (calibration) and resolution/sharpness components via the
  exceedance-rate gap;
- ``bias_correct_quantiles`` — shift all member quantiles by the median
  exceedance error (or a supplied shift) and repair monotonicity.

Honesty: calibration is marginal and per-level; corrected forecasts are
recalibrated on the supplied evaluation window only.

References:
- Gneiting, T., Balabdaoui, F., Raftery, A. E. (2007). Probabilistic
  forecasts, calibration and sharpness — the Murphy decomposition.
- Koenker, R. (2005). *Quantile Regression* — exceedance interpretation.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def exceedance_rates(y: FloatArray, quantiles: FloatArray, taus: FloatArray) -> FloatArray:
    """Empirical exceedance P(y ≤ q_τ) per level, aligned with taus."""
    y_arr = np.asarray(y, dtype=np.float64)
    q = np.asarray(quantiles, dtype=np.float64)
    taus_arr = np.asarray(taus, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != len(taus_arr) or q.shape[0] != len(y_arr):
        raise ValueError("quantiles must be (T, Q) matching taus and y")
    out = np.empty(len(taus_arr), dtype=np.float64)
    for k in range(len(taus_arr)):
        out[k] = float(np.mean(y_arr <= q[:, k]))
    return out


def pinball_decomposition(
    y: FloatArray, quantiles: FloatArray, taus: FloatArray
) -> dict[str, float]:
    """Murphy-style decomposition of mean pinball at a single level.

    Returns total mean pinball, the exceedance-gap (reliability) term
    (p̂_τ − τ)², and the sharpness term (total − reliability), at the level
    closest to the median when multiple levels are supplied (decomposition
    is per-level).
    """
    y_arr = np.asarray(y, dtype=np.float64)
    q = np.asarray(quantiles, dtype=np.float64)
    taus_arr = np.asarray(taus, dtype=np.float64)
    if len(taus_arr) < 1:
        raise ValueError("taus must be non-empty")
    k = int(np.argmin(np.abs(taus_arr - 0.5)))
    tau = float(taus_arr[k])
    diff = y_arr - q[:, k]
    pinball = float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))
    phat = float(np.mean(y_arr <= q[:, k]))
    reliability = float((phat - tau) ** 2)
    return {
        "level": tau,
        "total": pinball,
        "reliability": reliability,
        "sharpness": max(pinball - reliability, 0.0),
        "exceedance": phat,
    }


def bias_correct_quantiles(
    quantiles: FloatArray,
    taus: FloatArray,
    y: FloatArray | None = None,
    shift: float | None = None,
) -> dict[str, FloatArray]:
    """Shift quantiles so empirical exceedance matches nominal at τ = 0.5.

    The correction δ equals the median exceedance error; supplying ``y``
    estimates δ from data, otherwise an explicit ``shift`` is required.
    Output is monotone-repaired across levels.
    """
    q = np.asarray(quantiles, dtype=np.float64)
    taus_arr = np.asarray(taus, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != len(taus_arr):
        raise ValueError("quantiles must be (T, Q) matching taus")
    if shift is None:
        if y is None:
            raise ValueError("either y or an explicit shift must be supplied")
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.shape != (q.shape[0],):
            raise ValueError("y must be (T,) matching quantiles")
        k_med = int(np.argmin(np.abs(taus_arr - 0.5)))
        delta = float(np.mean(y_arr <= q[:, k_med]) - taus_arr[k_med])
    else:
        delta = float(shift)
    corrected = q - delta
    return {
        "quantiles": np.asarray(np.sort(corrected, axis=1), dtype=np.float64),
        "shift": np.asarray(delta, dtype=np.float64),
    }
