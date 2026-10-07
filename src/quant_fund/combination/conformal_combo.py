"""Split-conformal wrapper for ensemble point forecasts.

Wraps any combined point forecast (e.g. a member average) in marginal
prediction intervals with a distribution-free coverage guarantee: on fresh
exchangeable data the empirical coverage is at least 1 − α in expectation.
Use it to report honest intervals around an ensemble instead of an
assumed-Gaussian band.

Honesty: coverage is a marginal, finite-sample guarantee under the
exchangeability assumption of split conformal prediction; it is not a
conditional-coverage claim.

References:
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., Wasserman, L.
  (2018). Distribution-free predictive inference for regression — split
  conformal coverage.
- Papadopoulos, H., Proedrou, K., Vovk, V., Gammerman, A. (2002). Inductive
  confidence machines for regression — the original split conformal.
- Shafer, G., Vovk, V. (2008). A tutorial on conformal prediction.

Composition: pure numpy; deterministic quantiles (no randomness).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def split_conformal_intervals(
    cal_scores: FloatArray,
    pred_test: FloatArray,
    alpha: float,
) -> dict[str, FloatArray]:
    """Symmetric intervals pred ± q where q calibrates the score quantile.

    ``cal_scores`` are conformity scores (e.g. |y − pred|) on the held-out
    calibration set; the quantile index uses the finite-sample correction
    ceil((n+1)(1−α))/n so that coverage ≥ 1 − α holds marginally.
    """
    s = np.asarray(cal_scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    if len(s) < 2:
        raise ValueError("need at least 2 calibration scores")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    n = len(s)
    level = 1.0 - alpha
    k = int(np.ceil((n + 1) * level))
    k = min(max(k, 1), n)
    q = float(np.sort(s)[k - 1])
    pred = np.asarray(pred_test, dtype=np.float64)
    return {
        "lower": np.asarray(pred - q, dtype=np.float64),
        "upper": np.asarray(pred + q, dtype=np.float64),
        "q": np.asarray(q, dtype=np.float64),
        "alpha": np.asarray(alpha, dtype=np.float64),
    }


def empirical_coverage(y: FloatArray, lower: FloatArray, upper: FloatArray) -> float:
    """Fraction of observations inside [lower, upper] (NaNs excluded)."""
    y = np.asarray(y, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if y.shape != lo.shape or y.shape != hi.shape:
        raise ValueError("y, lower and upper must have equal shapes")
    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(mask):
        raise ValueError("no finite observations to score")
    inside = (y[mask] >= lo[mask]) & (y[mask] <= hi[mask])
    return float(np.mean(inside))
