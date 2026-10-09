"""Conformalized quantile regression (CQR) for member ensembles.

Wraps any member quantile forecasts in a distribution-free coverage
correction (Romano, Patterson & Candès 2019): conformity scores on a
calibration set adjust each side of every member band so the marginal
coverage of the corrected band is at least 1 − α.

- ``cqr_scores`` — side scores: max(q_lo − y, 0) for the lower side and
  max(y − q_hi, 0) for the upper side;
- ``cqr_correct`` — finite-sample-corrected per-side shifts from the
  calibration scores;
- ``cqr_intervals`` — corrected intervals on new predictions;
- ``cqr_coverage`` — empirical coverage of the corrected band.

Honesty: the coverage guarantee is marginal, finite-sample, and assumes
exchangeability between calibration and evaluation data.

References:
- Romano, Y., Patterson, E., Candès, E. J. (2019). Conformalized quantile
  regression — the CQR construction.
- Lei, J. et al. (2018). Distribution-free predictive inference for
  regression — the split-conformal machinery.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def cqr_scores(y: FloatArray, lo: FloatArray, hi: FloatArray) -> dict[str, FloatArray]:
    """CQR conformity scores per side (non-negative)."""
    y_arr = np.asarray(y, dtype=np.float64)
    lo_arr = np.asarray(lo, dtype=np.float64)
    hi_arr = np.asarray(hi, dtype=np.float64)
    if not (y_arr.shape == lo_arr.shape == hi_arr.shape):
        raise ValueError("y, lo and hi must have equal shapes")
    return {
        "lower": np.asarray(np.maximum(lo_arr - y_arr, 0.0), dtype=np.float64),
        "upper": np.asarray(np.maximum(y_arr - hi_arr, 0.0), dtype=np.float64),
    }


def _quantile_with_correction(scores: FloatArray, alpha: float) -> float:
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(s)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1])


def cqr_correct(
    y_cal: FloatArray,
    lo_cal: FloatArray,
    hi_cal: FloatArray,
    alpha: float,
) -> dict[str, float]:
    """Per-side finite-sample correction shifts from the calibration set.

    Each side is corrected at the supplied level independently, so a
    two-sided 1 − α band needs alpha = α/2 here (Bonferroni across sides);
    the split-conformal finite-sample bound then gives marginal coverage
    at least 1 − α on exchangeable evaluation data.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    sc = cqr_scores(y_cal, lo_cal, hi_cal)
    return {
        "lower_shift": _quantile_with_correction(sc["lower"], alpha),
        "upper_shift": _quantile_with_correction(sc["upper"], alpha),
    }


def cqr_intervals(
    lo_new: FloatArray,
    hi_new: FloatArray,
    correction: dict[str, float],
) -> dict[str, FloatArray]:
    """Apply the corrections: widen the lower bound downward, upper upward."""
    lo = np.asarray(lo_new, dtype=np.float64) - float(correction["lower_shift"])
    hi = np.asarray(hi_new, dtype=np.float64) + float(correction["upper_shift"])
    return {"lower": np.asarray(lo, dtype=np.float64), "upper": np.asarray(hi, dtype=np.float64)}


def cqr_coverage(y: FloatArray, lower: FloatArray, upper: FloatArray) -> float:
    """Empirical coverage of the corrected band (NaN-aware)."""
    y_arr = np.asarray(y, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if not (y_arr.shape == lo.shape == hi.shape):
        raise ValueError("y, lower and upper must have equal shapes")
    mask = np.isfinite(y_arr) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(mask):
        raise ValueError("no finite observations to score")
    return float(np.mean((y_arr[mask] >= lo[mask]) & (y_arr[mask] <= hi[mask])))
