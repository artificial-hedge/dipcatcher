"""Cost-aware optimal holding-period selection.

Turns the horizon-IC curve plus a turnover profile into a decision: which
holding lag maximises net predictive content after a linear cost drag.

- ``net_ic_curve`` — ic(k) − cost · turnover(k) per lag;
- ``optimal_holding_lag`` — argmax_k net_ic_curve with tie-break to the
  shorter lag; returns the full profile for inspection;
- ``lag_breakeven_table`` — per-lag cost level at which that lag's net IC
  is zero (inf when the raw IC is non-positive).

Honesty: cost units are abstract (IC points per unit turnover); the
"optimal" lag is optimal within the supplied curve, not a guarantee.

References:
- Grinold, R., Kahn, R. (2000). *Active Portfolio Management* — the
  fundamental law with costs and the horizon decision.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*,
  ch. 13 — capacity and turnover reasoning.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def net_ic_curve(ic: FloatArray, turnover: FloatArray, cost: float) -> FloatArray:
    """Net IC per lag: ic(k) − cost · turnover(k)."""
    ic = np.asarray(ic, dtype=np.float64)
    to = np.asarray(turnover, dtype=np.float64)
    if ic.shape != to.shape or ic.ndim != 1:
        raise ValueError("ic and turnover must be one-dimensional arrays of equal shape")
    if cost < 0:
        raise ValueError("cost must be non-negative")
    return np.asarray(ic - cost * to, dtype=np.float64)


def optimal_holding_lag(
    ic: FloatArray, turnover: FloatArray, cost: float
) -> dict[str, float | FloatArray]:
    """Argmax of the net IC curve with first-max tie-break.

    Returns the 1-based optimal lag, its net and gross IC, and the full
    net curve for downstream inspection.
    """
    net = net_ic_curve(ic, turnover, cost)
    if np.all(~np.isfinite(net)):
        raise ValueError("net IC curve is all-NaN")
    k = int(np.nanargmax(net)) + 1
    return {
        "lag": float(k),
        "net_ic": float(net[k - 1]),
        "gross_ic": float(np.asarray(ic, dtype=np.float64)[k - 1]),
        "curve": net,
    }


def lag_breakeven_table(ic: FloatArray, turnover: FloatArray) -> FloatArray:
    """Per-lag breakeven cost ic(k)/turnover(k); inf for non-positive IC."""
    ic = np.asarray(ic, dtype=np.float64)
    to = np.asarray(turnover, dtype=np.float64)
    if ic.shape != to.shape or ic.ndim != 1:
        raise ValueError("ic and turnover must be one-dimensional arrays of equal shape")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((ic > 0) & (to > 0), ic / np.maximum(to, 1e-12), np.inf)
    return np.asarray(out, dtype=np.float64)
