"""IC/turnover efficiency frontier across holding horizons.

Each holding horizon maps a signal to a point (mean turnover, mean IC). The
non-dominated set across horizons is the efficiency frontier: horizons on
it offer the best predictive content for their cost profile.

- ``ic_turnover_points`` — per-horizon (mean turnover, mean IC, net IC at a
  reference cost) tuples from horizon panels;
- ``efficiency_frontier`` — the set of non-dominated horizons (higher IC
  for lower turnover), in order of increasing turnover;
- ``trade_off_slope`` — local slope ΔIC/Δturnover along the frontier,
  measuring the price of a faster signal in IC units.

Honesty: the frontier is descriptive of the supplied evaluation window; it
is not a guarantee of out-of-sample net performance.

References:
- Grinold, R., Kahn, R. (2000). *Active Portfolio Management* — the
  cost-adjusted frontier.
- Zadeh, L. (1963). On the definition of adaptivity — Pareto dominance.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.decay.ic_series import spearman_ic
from quant_fund.decay.turnover import signal_turnover, signal_weights

FloatArray = NDArray[np.float64]


def ic_turnover_points(signals: list[FloatArray], actual: FloatArray) -> dict[str, FloatArray]:
    """Per-horizon (mean turnover, mean IC, IR) points.

    Each signal panel is (T, N); weights derive from the signal itself via
    ``signal_weights`` (gross=1). Returns aligned arrays per horizon.
    """
    if not signals:
        raise ValueError("signals must be non-empty")
    a = np.asarray(actual, dtype=np.float64)
    n_horizons = len(signals)
    to = np.empty(n_horizons, dtype=np.float64)
    ic = np.empty(n_horizons, dtype=np.float64)
    ir = np.empty(n_horizons, dtype=np.float64)
    for k, s in enumerate(signals):
        s_arr = np.asarray(s, dtype=np.float64)
        if s_arr.shape != a.shape:
            raise ValueError("each signal must match actual's shape")
        ics = spearman_ic(s_arr, a)
        ic[k] = float(np.nanmean(ics))
        ir[k] = float(ic[k] / max(float(np.nanstd(ics, ddof=1)), 1e-12))
        to[k] = float(np.nanmean(signal_turnover(signal_weights(s_arr))))
    return {"turnover": to, "ic": ic, "ir": ir}


def efficiency_frontier(turnover: FloatArray, ic: FloatArray) -> NDArray[np.int64]:
    """Indices of non-dominated horizons, ordered by increasing turnover.

    A horizon is dominated if another has strictly lower turnover with
    ≥ IC (or strictly higher IC with ≤ turnover).
    """
    to = np.asarray(turnover, dtype=np.float64)
    ic = np.asarray(ic, dtype=np.float64)
    if to.shape != ic.shape or to.ndim != 1:
        raise ValueError("turnover and ic must be one-dimensional arrays of equal shape")
    order = np.argsort(to)
    frontier = []
    best_ic = -np.inf
    for i in order:
        if ic[i] > best_ic:
            frontier.append(int(i))
            best_ic = float(ic[i])
    return np.asarray(frontier, dtype=np.int64)


def trade_off_slope(turnover: FloatArray, ic: FloatArray) -> FloatArray:
    """ΔIC / Δturnover along the frontier (between consecutive members)."""
    to = np.asarray(turnover, dtype=np.float64)
    ic = np.asarray(ic, dtype=np.float64)
    if to.shape != ic.shape or to.ndim != 1:
        raise ValueError("turnover and ic must be one-dimensional arrays of equal shape")
    frontier = efficiency_frontier(to, ic)
    if len(frontier) < 2:
        return np.zeros(0, dtype=np.float64)
    dto = np.diff(to[frontier])
    dic = np.diff(ic[frontier])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(dto) > 1e-12, dic / dto, np.nan)
    return np.asarray(out, dtype=np.float64)
