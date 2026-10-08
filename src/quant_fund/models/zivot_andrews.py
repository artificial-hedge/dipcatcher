"""Zivot-Andrews (1992) endogenous-break unit-root test.

References
----------
- Zivot, E. & Andrews, D.W.K. (1992). "Further Evidence on
  the Great Crash, the Oil-Price Shock, and the Unit-Root
  Hypothesis." *Journal of Business & Economic Statistics*
  10(3), 251-270.
- Perron, P. (1989). "The Great Crash, the Oil Price Shock,
  and the Unit Root Hypothesis." *Econometrica* 57(6),
  1361-1401.
- Banerjee, A., Lumsdaine, R.L. & Stock, J.H. (1992).
  "Recursive and Sequential Tests of the Unit-Root and
  Trend-Break Hypotheses." *Journal of Business & Economic
  Statistics* 10(3), 271-287.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
The endogenous-break extension of Perron (1989): allow one
unknown break in the deterministic component and take the
*least favorable* evidence for stationarity — ``t_min`` over
a grid of candidate break dates ``lambda*T``. Three models:
A (level break), B (trend break), C (both). We implement
model C (the most-quoted variant): for each candidate
break ``b``, estimate

    dy = a + pi*y_{t-1} + theta*DU_t(b) + gamma*DT_t(b)
         + sum phi_j dy_{t-j} + e,

where ``DU = 1[t > b]`` (level dummy) and ``DT = t - b
times DU`` (trend break), then record the DF t-statistic
on ``pi``; the break is the argmin over ``b in [0.1T, 0.9T]``.
Critical values are *more negative* than standard DF
because minimization cherry-picks — we use the Zivot-Andrews
Table model-C 5% value -5.08 (vs -3.45 conventional): a
series that is a stationary AR around a broken trend will
beat ordinary DF but still trip the ZA bar. The synth
plants a level break mid-sample in an AR(0.4) series
(reject unit root, recover break date) vs a plain RW.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def zivot_andrews(
    x: FloatArray,
    k: int = 1,
    trim: float = 0.15,
) -> dict[str, float]:
    """ZA model-C (level+trend break) minimum-t test."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 1 or xx.size < 120 or not np.all(np.isfinite(xx)):
        raise ValueError("bad series")
    if np.std(np.diff(xx)) < 1e-12:
        raise ValueError("degenerate")
    t = xx.size
    dy = np.diff(xx)
    best_t, best_b = np.inf, -1
    lo, hi = int(t * trim), int(t * (1 - trim))
    for b in range(lo, hi):
        rows = []
        # regress dy_i on const, y_{i-1}, DU_i(b), DT_i(b), lagged dy
        for i in range(k, dy.size):
            tt = i + 1
            du = 1.0 if tt > b else 0.0
            dt = (tt - b) * du
            row = [1.0, xx[i], du, dt] + [dy[i - j] for j in range(1, k + 1)]
            rows.append(row)
        xm = np.asarray(rows)
        target = dy[k:]
        coef, *_ = np.linalg.lstsq(xm, target, rcond=None)
        resid = target - xm @ coef
        dof = resid.size - xm.shape[1]
        if dof < 10:
            continue
        s2 = float(resid @ resid / dof)
        cov = s2 * np.linalg.pinv(xm.T @ xm)
        se = float(np.sqrt(max(cov[1, 1], 1e-30)))
        tstat = float(coef[1] / se)
        if tstat < best_t:
            best_t, best_b = tstat, b
    return {
        "t_min": float(best_t),
        "break_idx": float(best_b),
        "crit5": -5.08,
        "reject_unit_root": float(best_t < -5.08),
        "lag": float(k),
    }


def synth_za(
    seed: int = 20261231 + 328,
    n: int = 350,
    break_frac: float = 0.5,
) -> tuple[FloatArray, FloatArray, int]:
    """SYNTHETIC AR(0.4) w/ level break vs plain RW."""
    rng = np.random.default_rng(seed)
    b = int(n * break_frac)
    st = np.zeros(n)
    st[b:] = 3.0  # level shift
    for t in range(1, n):
        st[t] += 0.4 * st[t - 1] + rng.normal()
    rw = np.cumsum(rng.normal(0.0, 1.0, n))
    return np.asarray(st), np.asarray(rw), b


def bench_zivot_andrews(
    seed: int = 20261231 + 328,
) -> dict[str, float]:
    """Wave-56 self-check: break AR rejected, break recovered."""
    st, rw, b = synth_za(seed=seed)
    r_st = zivot_andrews(st)
    r_rw = zivot_andrews(rw)
    ok = (
        r_st["reject_unit_root"] == 1.0
        and r_rw["reject_unit_root"] == 0.0
        and abs(r_st["break_idx"] - b) < 60
    )
    return {
        "synthetic_t_min_st": r_st["t_min"],
        "synthetic_t_min_rw": r_rw["t_min"],
        "synthetic_break_hat": r_st["break_idx"],
        "synthetic_break_true": float(b),
        "synthetic_score": float(ok),
    }
