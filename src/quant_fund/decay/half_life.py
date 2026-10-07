"""Signal half-life estimation.

Two complementary estimators:

- exponential fit: log |IC(k)| regressed on lag k (WLS), half-life =
  ln 2 / decay rate — measures how quickly *predictive content* fades;
- AR(1) half-life: fit x_t = φ x_{t−1} + ε on a stationary series (e.g. the
  IC series itself or an alpha exposure), half-life = −ln 2 / ln φ —
  measures how quickly the *signal itself* mean-reverts.

Both return NaN guards when the estimated process does not decay. The AR(1)
estimator on a near-unit-root series returns a long but finite half-life —
the documented behaviour of OLS on persistent series.

Honesty: half-lives describe the supplied sample's autocorrelation/predictive
decay structure. They are not stable-horizon guarantees.

References:
- López de Prado, M. (2018). *Advances in Financial Machine Learning*,
  ch. 5 — signal horizon and fractional-decay reasoning.
- Box, G. E. P., Jenkins, G. M. (1970). *Time Series Analysis* — AR(1)
  autocorrelation and half-life.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _ls(x: FloatArray, y: FloatArray) -> tuple[float, float]:
    """Simple OLS slope/intercept of y on x (returns slope, intercept)."""
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    sxx = float(np.sum((x - xm) ** 2))
    if sxx <= 0:
        return float("nan"), float("nan")
    slope = float(np.sum((x - xm) * (y - ym)) / sxx)
    return slope, float(ym - slope * xm)


def fit_exponential_decay(lags: FloatArray, values: FloatArray) -> dict[str, float]:
    """WLS fit of log|value| = a + b·lag; half-life = ln2 / (−b) for b < 0."""
    lags = np.asarray(lags, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if lags.shape != values.shape or lags.ndim != 1:
        raise ValueError("lags and values must be one-dimensional arrays of equal shape")
    mask = np.isfinite(values) & (np.abs(values) > 0) & np.isfinite(lags)
    x = lags[mask]
    y = np.log(np.abs(values[mask]))
    if len(x) < 3:
        raise ValueError("need at least 3 non-zero decay points")
    slope, intercept = _ls(x, y)
    if not np.isfinite(slope) or slope >= 0:
        return {
            "decay_rate": float(slope) if np.isfinite(slope) else float("nan"),
            "half_life": float("inf"),
            "intercept": float(intercept),
            "r2": float("nan"),
            "n": float(len(x)),
        }
    fitted = intercept + slope * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "decay_rate": slope,
        "half_life": float(np.log(2.0) / (-slope)),
        "intercept": float(intercept),
        "r2": r2,
        "n": float(len(x)),
    }


def ar1_half_life(x: FloatArray) -> dict[str, float]:
    """AR(1) half-life of a stationary series (OLS with intercept)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        raise ValueError("need at least 10 observations")
    lag_x = x[:-1]
    cur_x = x[1:]
    phi, _ = _ls(lag_x, cur_x)
    if not np.isfinite(phi) or abs(phi) >= 1.0 or phi <= 0.0:
        return {"phi": float(phi) if np.isfinite(phi) else float("nan"), "half_life": float("inf")}
    return {"phi": phi, "half_life": float(-np.log(2.0) / np.log(phi))}


def classify_decay(half_life: float, fast: float = 2.0, slow: float = 5.0) -> str:
    """Bucket a half-life into fast / medium / slow (NaN → unknown)."""
    if not np.isfinite(half_life):
        return "unknown"
    if half_life < fast:
        return "fast"
    if half_life < slow:
        return "medium"
    return "slow"
