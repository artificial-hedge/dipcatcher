"""Mean-shift segmentation shared by statistical and event-time diagnostics.

Scott–Knott binary segmentation minimizes within-segment squared deviations;
these primitives have no model fitting, simulator, or orchestration dependency.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from quant_fund.utils.series import finite_observations

Array = NDArray[np.float64]
IdxArray = NDArray[np.intp]


def _as_vector(x: Array, name: str = "x", *, min_obs: int = 8) -> Array:
    return finite_observations(x, name, min_obs=min_obs)


def _seg_cost(v: Array, a: int, b: int) -> float:
    """Within-segment cost = sum of squared deviations (Gaussian mean-shift)."""
    seg = v[a:b]
    if seg.size <= 0:
        return 0.0
    d = seg - seg.mean()
    return float(d @ d)


def binary_segmentation(
    x: Array, min_size: int = 10, penalty: float | None = None, max_cps: int = 10
) -> IdxArray:
    """Scott–Knott (1974) binary segmentation on mean shifts.

    Recursively splits at the argmax CUSUM split point while the split reduces
    total cost by more than ``penalty`` (default ``2 * log(n) * var`` — a BIC
    analogue).  Returns sorted changepoint indices.
    """
    v = _as_vector(x)
    if isinstance(min_size, bool) or not isinstance(min_size, int) or min_size < 2:
        raise ValueError("min_size must be an integer >= 2")
    var = float(v.var(ddof=1))
    pen = 2.0 * math.log(v.size) * max(var, 1e-12) * 2 if penalty is None else float(penalty)
    if not np.isfinite(pen) or pen < 0.0:
        raise ValueError("penalty must be non-negative and finite")
    cps: list[int] = []

    def _split(a: int, b: int) -> None:
        if len(cps) >= max_cps or b - a < 2 * min_size:
            return
        best_gain, best_t = 0.0, -1
        base = _seg_cost(v, a, b)
        for t in range(a + min_size, b - min_size + 1):
            gain = base - _seg_cost(v, a, t) - _seg_cost(v, t, b)
            if gain > best_gain:
                best_gain, best_t = gain, t
        if best_t > 0 and best_gain > pen:
            cps.append(best_t)
            _split(a, best_t)
            _split(best_t, b)

    _split(0, v.size)
    return np.asarray(sorted(cps), dtype=np.intp)
