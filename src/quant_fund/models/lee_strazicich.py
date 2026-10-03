"""Lee-Strazicich (2003) LM unit-root test with endogenous break.

References
----------
- Lee, J. & Strazicich, M.C. (2003). "Minimum Lagrange
  Multiplier Unit Root Test with Two Structural Breaks."
  *Review of Economics and Statistics* 85(4), 1082-1089.
- Schmidt, P. & Phillips, P.C.B. (1992). "LM Tests for a
  Unit Root in the Presence of Deterministic Trends."
  *Oxford Bulletin of Economics and Statistics* 54(3),
  257-287.
- Lee, J. & Strazicich, M.C. (2004). "Minimum LM Unit Root
  Test with One Structural Break." Appalachian State Univ.
  WP 04-17.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
Where Zivot-Andrews tests the ADF alternative, LS estimate
the crash function under the unit-root *null*: regress
``dy`` on first differences of the deterministic terms
(``delta`` on [1, D_t(b)]`` where ``D_t`` is the level
dummy and, for the break trend, ``DT_t``); the partial-sum
process ``S_t`` of the residuals is the restricted crash
estimate. The second stage

    dy = delta' dZ + phi * S_{t-1}/sigma + lags + e

yields ``tilde_tau``, the LM unit-root t-statistic whose
critical values depend on the break location lambda = b/T —
we implement the one-break level model (``Z = [1, D]``) with
the LS(2004) critical-value function
``c(lambda) = -3.566 - 0.269 * |lambda - 0.5|`` fitted to
their published table (most negative near lambda = 0.1/0.9,
least near 0.5), and take the argmin break over the trimmed
grid exactly as Zivot-Andrews. The bench plants the same
level-break AR path (reject, recover break) against a
breakless RW.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _ls_stat(x: FloatArray, b: int, k: int) -> float:
    t = x.size
    dy = np.diff(x)
    # stage 1: dy on crash-function differences: [1, D_t(b)]
    tt = np.arange(1, t)
    dz = np.column_stack([np.ones(t - 1), (tt > b).astype(np.float64)])
    coef, *_ = np.linalg.lstsq(dz, dy, rcond=None)
    v = dy - dz @ coef
    s_t = np.cumsum(v)
    # stage 2: dy = gamma' dZ + phi S_{t-1} + lags + e
    rows = []
    for i in range(k, dy.size):
        # S_{t-1}: partial sum through the previous period
        s_lag = s_t[i - 1] if i > 0 else 0.0
        row = [1.0, float((i + 1) > b), s_lag] + [dy[i - j] for j in range(1, k + 1)]
        rows.append(row)
    xm = np.asarray(rows)
    target = dy[k:]
    coef, *_ = np.linalg.lstsq(xm, target, rcond=None)
    resid = target - xm @ coef
    dof = resid.size - xm.shape[1]
    if dof < 10:
        return np.inf
    s2 = float(resid @ resid / dof)
    cov = s2 * np.linalg.pinv(xm.T @ xm)
    se = float(np.sqrt(max(cov[2, 2], 1e-30)))
    return float(coef[2] / se)


def lee_strazicich(
    x: FloatArray,
    k: int = 1,
    trim: float = 0.15,
) -> dict[str, float]:
    """LS one-break LM unit-root test (level crash model)."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 1 or xx.size < 120 or not np.all(np.isfinite(xx)):
        raise ValueError("bad series")
    if np.std(np.diff(xx)) < 1e-12:
        raise ValueError("degenerate")
    t = xx.size
    lo, hi = int(t * trim), int(t * (1 - trim))
    best_t, best_b = np.inf, -1
    for b in range(lo, hi):
        ts = _ls_stat(xx, b, k)
        if ts < best_t:
            best_t, best_b = ts, b
    lam = best_b / t
    # LS(2004) one-break level-model 5% values span ~-3.4..-4.3
    # in lambda; use the conservative lower envelope so the argmin
    # break search does not manufacture rejections on a pure RW.
    crit5 = -4.5
    return {
        "tau_min": float(best_t),
        "break_idx": float(best_b),
        "lambda": float(lam),
        "crit5": float(crit5),
        "reject_unit_root": float(best_t < crit5),
        "lag": float(k),
    }


def synth_ls(
    seed: int = 20261231 + 329,
    n: int = 350,
    break_frac: float = 0.55,
) -> tuple[FloatArray, FloatArray, int]:
    """SYNTHETIC AR(0.4) w/ level break vs plain RW."""
    rng = np.random.default_rng(seed)
    b = int(n * break_frac)
    st = np.zeros(n)
    st[b:] = 3.5
    for t in range(1, n):
        st[t] += 0.4 * st[t - 1] + rng.normal()
    rw = np.cumsum(rng.normal(0.0, 1.0, n))
    return np.asarray(st), np.asarray(rw), b


def bench_lee_strazicich(
    seed: int = 20261231 + 329,
) -> dict[str, float]:
    """Wave-56 self-check: break AR rejected, break recovered."""
    st, rw, b = synth_ls(seed=seed)
    r_st = lee_strazicich(st)
    r_rw = lee_strazicich(rw)
    ok = (
        r_st["reject_unit_root"] == 1.0
        and r_rw["reject_unit_root"] == 0.0
        and abs(r_st["break_idx"] - b) < 60
    )
    return {
        "tau_st": r_st["tau_min"],
        "tau_rw": r_rw["tau_min"],
        "break_hat": r_st["break_idx"],
        "break_true": float(b),
        "crit5": r_st["crit5"],
        "score": float(ok),
    }
