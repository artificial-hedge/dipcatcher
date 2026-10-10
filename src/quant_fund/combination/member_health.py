"""Member-level ensemble health diagnostics.

Beyond combination weights, you want per-member quality answers:

- ``member_pinball`` — per-member mean pinball on the supplied panel;
- ``leave_one_out_contribution`` — the change in ensemble pinball when each
  member is dropped (positive = member helps, negative = hurts);
- ``member_loss_drift`` — CUSUM-style detector on a member's rolling loss
  series to flag a member that has degraded;
- ``member_health_report`` — the bundle, with a per-member "healthy" flag
  against a tolerance.

Honesty: health flags are descriptive on the supplied evaluation window;
they do not decide production membership.

References:
- Caruana, R. et al. (2004). Ensemble selection from libraries of models —
  leave-one-out contribution reasoning.
- Page, E. S. (1954). Continuous inspection schemes — the CUSUM drift
  detector.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _mean_pinball_matrix(q: FloatArray, y: FloatArray, taus: FloatArray) -> FloatArray:
    """(T, M) pinball per member averaged over quantile levels."""
    diff = y[:, None, None] - q
    loss = np.maximum(taus[None, None, :] * diff, (taus[None, None, :] - 1.0) * diff)
    return np.asarray(loss.mean(axis=2), dtype=np.float64)


def member_pinball(member_quantiles: FloatArray, y: FloatArray, taus: FloatArray) -> FloatArray:
    """Per-member mean pinball over the whole panel."""
    q = np.asarray(member_quantiles, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    taus_arr = np.asarray(taus, dtype=np.float64)
    if q.ndim != 3 or y_arr.shape != (q.shape[0],) or taus_arr.shape != (q.shape[2],):
        raise ValueError("member_quantiles (T, M, Q), y (T,), taus (Q,) must align")
    return np.asarray(_mean_pinball_matrix(q, y_arr, taus_arr).mean(axis=0), dtype=np.float64)


def leave_one_out_contribution(
    member_quantiles: FloatArray,
    y: FloatArray,
    taus: FloatArray,
) -> dict[str, FloatArray]:
    """Δ pinball when dropping each member from the equal-weight ensemble.

    contribution = loss(ensemble without k) − loss(full ensemble):
    positive = the member helps (dropping raises loss); negative = the
    member hurts (dropping lowers loss). Note that a member with poor
    standalone pinball can still help through diversification.
    """
    q = np.asarray(member_quantiles, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    taus_arr = np.asarray(taus, dtype=np.float64)
    if q.ndim != 3 or y_arr.shape != (q.shape[0],) or taus_arr.shape != (q.shape[2],):
        raise ValueError("member_quantiles (T, M, Q), y (T,), taus (Q,) must align")
    m = q.shape[1]
    full = q.mean(axis=1)
    loss_full = _mean_pinball_matrix(full[:, None, :], y_arr, taus_arr).mean(axis=0)
    contrib = np.empty(m, dtype=np.float64)
    for k in range(m):
        keep = [j for j in range(m) if j != k]
        sub = q[:, keep, :].mean(axis=1)
        loss_sub = _mean_pinball_matrix(sub[:, None, :], y_arr, taus_arr).mean(axis=0)[0]
        contrib[k] = float(loss_sub - loss_full[0])
    return {"contribution": np.asarray(contrib, dtype=np.float64)}


def member_loss_drift(
    member_quantiles: FloatArray,
    y: FloatArray,
    taus: FloatArray,
    *,
    pre_period: int = 50,
    threshold: float = 5.0,
    slack: float = 1.0,
) -> dict[str, FloatArray]:
    """Tabular CUSUM alarm per member on its rolling mean pinball.

    S_t = max(0, S_{t−1} + (pinball_t − μ₀)/σ₀ − δ) with slack δ — the
    reference shift the detector is tuned to catch. Alarms at the first t
    with S_t > threshold. Centre/scale are estimated robustly
    (median/MAD) because pinball loss series are skewed; without the
    robust scale a few heavy-tail spikes drive false alarms on long
    stable series. The slack keeps long stable series quiet (no drift →
    S decays instead of accumulating noise).
    """
    q = np.asarray(member_quantiles, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    taus_arr = np.asarray(taus, dtype=np.float64)
    if q.ndim != 3 or y_arr.shape != (q.shape[0],) or taus_arr.shape != (q.shape[2],):
        raise ValueError("member_quantiles (T, M, Q), y (T,), taus (Q,) must align")
    losses = _mean_pinball_matrix(q, y_arr, taus_arr)
    t_total, m = losses.shape
    if t_total < pre_period + 5:
        raise ValueError("series too short for the requested pre-period")
    alarm = np.full(m, -1.0)
    max_cusum = np.zeros(m, dtype=np.float64)
    for k in range(m):
        x = losses[:, k]
        pre = x[:pre_period]
        mu0 = float(np.median(pre))
        sd0 = float(1.4826 * np.median(np.abs(pre - mu0)))
        if sd0 <= 0:
            continue
        s = 0.0
        for t in range(pre_period, t_total):
            s = max(0.0, s + (x[t] - mu0) / sd0 - slack)
            max_cusum[k] = max(max_cusum[k], s)
            if s > threshold and alarm[k] < 0:
                alarm[k] = float(t)
    return {"alarm_time": alarm, "max_cusum": max_cusum}


def member_health_report(
    member_quantiles: FloatArray,
    y: FloatArray,
    taus: FloatArray,
    *,
    drift_threshold: float = 5.0,
    pre_period: int = 50,
    slack: float = 1.0,
) -> dict[str, FloatArray]:
    """Per-member pinball, leave-one-out contribution, drift alarm, flag."""
    pin = member_pinball(member_quantiles, y, taus)
    contrib = np.asarray(
        leave_one_out_contribution(member_quantiles, y, taus)["contribution"],
        dtype=np.float64,
    )
    drift = member_loss_drift(
        member_quantiles,
        y,
        taus,
        pre_period=pre_period,
        threshold=drift_threshold,
        slack=slack,
    )
    alarm = np.asarray(drift["alarm_time"], dtype=np.float64)
    healthy = (alarm < 0) & (contrib >= 0.0)
    return {
        "pinball": pin,
        "contribution": contrib,
        "drift_alarm": alarm,
        "healthy": healthy.astype(np.float64),
    }
