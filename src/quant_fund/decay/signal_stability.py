"""Rolling IC stability and structural-break diagnostics.

Tracks the signal's IC through rolling windows and flags when its
statistical behaviour changes:

- ``rolling_ic_stats`` — rolling mean/std/IR over a window with
  min-periods semantics (NaN until the window fills);
- ``cusum_break`` — CUSUM detector on the IC series: cumulative deviation
  from the pre-period mean scaled by pre-period std; alarms when the
  cumulative sum exceeds a threshold (default 5σ·√n normalisation);
- ``stability_score`` — fraction of rolling windows whose |IR| exceeds a
  reference level, in [0, 1].

Honesty: break detection flags a change point in the supplied sample; it
does not attribute a cause.

References:
- Page, E. S. (1954). Continuous inspection schemes — the CUSUM.
- Chu, C.-S. J., Stinchcombe, M., White, H. (1996). Monitoring structural
  change — CUSUM monitoring of time series.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _valid(x: FloatArray) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    return cast(FloatArray, arr[np.isfinite(arr)])


def rolling_ic_stats(
    ic: FloatArray, window: int = 60, min_periods: int = 20
) -> dict[str, FloatArray]:
    """Rolling mean/std/IR of the IC series with NaN warm-up."""
    x = np.asarray(ic, dtype=np.float64)
    if window < 2 or min_periods < 2 or min_periods > window:
        raise ValueError("need 2 <= min_periods <= window")
    n = len(x)
    mean = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    ir = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - window + 1)
        w = x[lo : t + 1]
        w = w[np.isfinite(w)]
        if len(w) >= min_periods:
            mean[t] = float(np.mean(w))
            sd[t] = float(np.std(w, ddof=1))
            ir[t] = float(mean[t] / sd[t]) if sd[t] > 0 else np.nan
    return {"mean": mean, "std": sd, "ir": ir}


def cusum_break(
    ic: FloatArray,
    *,
    pre_period: int = 50,
    threshold: float = 5.0,
) -> dict[str, float]:
    """CUSUM alarm on the IC series against its pre-period mean.

    S_t = max(0, S_{t−1} + (x_t − μ₀)/σ₀); alarms at the first t with
    S_t > threshold. Returns the alarm time (−1 if none), the max CUSUM,
    and the pre-period estimates used.
    """
    x = _valid(ic)
    n = len(x)
    if n < pre_period + 5:
        raise ValueError("series too short for the requested pre-period")
    mu0 = float(np.mean(x[:pre_period]))
    sd0 = float(np.std(x[:pre_period], ddof=1))
    if sd0 <= 0:
        raise ValueError("pre-period IC has no variation")
    s = 0.0
    s_max = 0.0
    alarm = -1.0
    for t in range(pre_period, n):
        s = max(0.0, s + (x[t] - mu0) / sd0)
        s_max = max(s_max, s)
        if s > threshold and alarm < 0:
            alarm = float(t)
    return {"alarm_time": alarm, "max_cusum": float(s_max), "mu0": mu0, "sd0": sd0}


def stability_score(
    ic: FloatArray, window: int = 60, min_periods: int = 20, ref_ir: float = 0.3
) -> float:
    """Fraction of filled rolling windows with |IR| >= ref_ir."""
    stats = rolling_ic_stats(ic, window=window, min_periods=min_periods)
    ir = stats["ir"]
    ir = ir[np.isfinite(ir)]
    if len(ir) == 0:
        raise ValueError("no filled rolling windows")
    return float(np.mean(np.abs(ir) >= ref_ir))
