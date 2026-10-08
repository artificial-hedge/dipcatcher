"""Giacomini-Rossi (2010) forecast-comparison fluctuation test.

References
----------
- Giacomini, R. & Rossi, B. (2010). "Forecast Comparisons in
  Unstable Environments." *Journal of Applied Econometrics*
  25(4), 595-620.
- Giacomini, R. & Rossi, B. (2009). "Detecting and Predicting
  Forecast Breakdowns." *Review of Economic Studies* 76(2),
  669-705.
- Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive
  Accuracy." *JBES* 13(3), 253-263.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
The GR fluctuation test monitors the out-of-sample relative
performance of two forecasters through time. With per-period
loss differential ``d_t = l_{1,t} - l_{2,t}``, a rolling window
of size ``m`` forms the standardized statistic
``tau_j = mean(d_{j:j+m}) / se`` evaluated at every start j;
under the null of stable relative performance, the sequence of
``tau`` stays within critical bands that depend on the window
share ``mu = m/T``. We implement the absolute-max variant with
a moving-block-bootstrap null: center ``d``, resample blocks of
expected length ``b``, and collect the distribution of
``max_j |tau_j|`` — the p-value is the bootstrap tail beyond the
observed maximum, and the breakdown location is ``argmax``. The
bench plants a stable differential (null, must accept) and a
half-sample degradation of forecaster 2 (alternative: rejects
with the detected breakdown near the midpoint).
``gr_fluctuation(d, m, block, n_boot, seed)`` returns the tau
path, ``max_stat``, ``p_boot``, ``break_idx``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _tau_path(d: FloatArray, m: int) -> FloatArray:
    """Rolling standardized mean differential, starts 0..T-m."""
    t = d.size
    if m < 20 or m >= t:
        raise ValueError("bad window")
    c = np.concatenate([[0.0], np.cumsum(d)])
    rm = (c[m:] - c[:-m]) / m
    se = float(np.std(d, ddof=1)) / np.sqrt(m)
    if se <= 0:
        raise ValueError("zero-variance differential")
    return np.asarray(rm / se, dtype=np.float64)


def gr_fluctuation(
    d: FloatArray,
    m: int,
    block: int = 10,
    n_boot: int = 400,
    seed: int = 0,
) -> dict[str, float | FloatArray]:
    """Fluctuation test on loss differential ``d`` (window ``m``)."""
    dd = np.asarray(d, dtype=np.float64)
    if dd.ndim != 1 or dd.size < 80 or not np.all(np.isfinite(dd)):
        raise ValueError("bad differential")
    tau = _tau_path(dd, m)
    obs = float(np.max(np.abs(tau)))
    t = dd.size
    rng = np.random.default_rng(seed)
    dc = dd - dd.mean()
    n_blocks = int(np.ceil(t / block))
    n_starts = max(t - block + 1, 1)
    stat = np.empty(n_boot)
    for r_i in range(n_boot):
        idx = np.empty(n_blocks * block, dtype=np.int64)
        starts = rng.integers(0, n_starts, size=n_blocks)
        for j, s in enumerate(starts):
            idx[j * block : (j + 1) * block] = np.arange(s, s + block)
        stat[r_i] = float(np.max(np.abs(_tau_path(dc[idx[:t]], m))))
    p = float(np.mean(stat >= obs))
    # breakdown onset: first start whose window tau breaches the
    # bootstrapped 95% band (argmax sits anywhere on the plateau)
    q95 = float(np.quantile(stat, 0.95))
    hits = np.flatnonzero(np.abs(tau) > q95)
    break_idx = int(hits[0]) if hits.size else -1
    return {
        "tau": tau,
        "max_stat": obs,
        "p_boot": p,
        "break_idx": float(break_idx),
        "crit_95": q95,
        "window": float(m),
    }


def synth_gr(
    seed: int = 20261231 + 312,
    t: int = 600,
    break_frac: float | None = None,
) -> FloatArray:
    """SYNTHETIC loss differential; optional mean shift at break_frac.

    Under the null the differential is a persistent AR(1) with
    zero mean; a breakdown doubles the mean after ``break_frac``.
    """
    rng = np.random.default_rng(seed)
    z = np.empty(t)
    z[0] = rng.standard_normal()
    for i in range(1, t):
        z[i] = 0.7 * z[i - 1] + rng.standard_normal()
    d = z / np.std(z)
    if break_frac is not None:
        k = int(t * break_frac)
        d[k:] += 2.0
    return d


def bench_giacomini_rossi(
    seed: int = 20261231 + 312,
) -> dict[str, float]:
    """Wave-54 self-check: stable accepted, breakdown rejected near mid."""
    d0 = synth_gr(seed=seed)
    r0 = gr_fluctuation(d0, m=100, n_boot=250, seed=seed)
    d1 = synth_gr(seed=seed + 5, break_frac=0.5)
    r1 = gr_fluctuation(d1, m=100, n_boot=250, seed=seed)
    ok = (
        float(r0["p_boot"]) > 0.1
        and float(r1["p_boot"]) < 0.05
        and abs(float(r1["break_idx"]) - 300) < 100
    )
    return {
        "synthetic_p_null": float(r0["p_boot"]),
        "synthetic_p_alt": float(r1["p_boot"]),
        "synthetic_break_idx": float(r1["break_idx"]),
        "synthetic_max_stat_alt": float(r1["max_stat"]),
        "synthetic_score": float(ok),
    }
