"""HEGY seasonal unit-root and Canova-Hansen seasonal-stability (SYNTHETIC)
tests.

Hylleberg, Engle, Granger & Yoo (1990) decompose the seasonal
difference 1 - B^s into roots at frequencies 0, pi, and the
harmonic pairs. For quarterly data (s = 4) the transformed
regressors are

    z1_t = (1 + B + B^2 + B^3) y_{t-1}      (frequency 0)
    z2_t = -(1 - B + B^2 - B^3) y_{t-1}     (frequency pi)
    z3_t = -(1 - B^2) y_{t-1}               (pi/2 pair, part 1)
    z4_t = -(1 - B^2) y_{t-2}               (pi/2 pair, part 2)

in the regression  Delta_4 y_t = pi1 z1 + pi2 z2 + pi3 z3 +
pi4 z4 + seasonal dummies + eps.  t(pi1), t(pi2) and the joint
F(pi3 = pi4 = 0) carry the unit-root evidence. Critical values
are nonstandard, so both tests here use internally simulated
null distributions (deterministic rng — honest Monte Carlo
p-values, documented as such rather than tabulated constants
passed off as exact).

Canova & Hansen (1995) test the stability of seasonal dummies:
regress y on seasonal dummies (+ intercept), accumulate the
scaled residual-sums per season, and compare

    L = (1/T^2) tr( S^{-1} M ),  M_j = sum_t (sum_{u<=t} e_u d_j,u)^2

against its null distribution under constant seasonality —
also simulated internally.

Honesty: quarterly s = 4 only (the canonical HEGY case);
autocorrelation in the residual is not augmented (no lag
terms — flagged as the plain HEGY regression); Monte Carlo
p-values have ~1/sqrt(mc) resolution. Fail-closed on n < 8s
or constant input.

References: Hylleberg, Engle, Granger & Yoo (1990) "Seasonal
integration and cointegration", J. Econometrics 44:215;
Ghysels, Lee & Noh (1994) J. Econometrics 62:415; Canova &
Hansen (1995) JBES 13:237; Osborn (1990) survey.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _hegy_regressors(y: FloatArray, s: int) -> tuple[FloatArray, FloatArray]:
    """Build (Delta_s y, [z1..z_{s} seasonal]) for s = 4.

    Returns the LHS and the z-matrix WITHOUT the deterministic
    block (seasonal dummies appended by caller).
    """
    t = y.size
    d4 = y[s:] - y[:-s]
    m = d4.size
    # z1 = y_{t-1}+y_{t-2}+y_{t-3}+y_{t-4} evaluated at t = s..t-1
    z1 = np.array([y[k - 1] + y[k - 2] + y[k - 3] + y[k - 4] for k in range(s, t)])
    # z2 = -(y_{t-1} - y_{t-2} + y_{t-3} - y_{t-4})
    z2 = np.array([-(y[k - 1] - y[k - 2] + y[k - 3] - y[k - 4]) for k in range(s, t)])
    # z3 = -(y_{t-1} - y_{t-3})
    z3 = np.array([-(y[k - 1] - y[k - 3]) for k in range(s, t)])
    # z4 = -(y_{t-2} - y_{t-4})
    z4 = np.array([-(y[k - 2] - y[k - 4]) for k in range(s, t)])
    z = np.column_stack([z1, z2, z3, z4])
    if not (z.shape[0] == m):
        raise ValueError("z.shape[0] == m")
    return np.asarray(d4, dtype=np.float64), np.asarray(z, dtype=np.float64)


def _seasonal_dummies(n: int, s: int) -> FloatArray:
    d = np.zeros((n, s))
    for i in range(s):
        d[i::s, i] = 1.0
    # drop one column to avoid collinearity with nothing else in
    # the block (all s dummies — no separate intercept)
    return d


def _hegy_stats(y: FloatArray, s: int) -> tuple[float, float, float]:
    """Return (t_pi1, t_pi2, F_pi34) for one series."""
    d4, z = _hegy_regressors(y, s)
    n = d4.size
    d = _seasonal_dummies(n, s)
    x = np.column_stack([d, z])
    k = x.shape[1]
    beta, *_ = np.linalg.lstsq(x, d4, rcond=None)
    e = d4 - x @ beta
    dof = max(n - k, 1)
    s2 = float(e @ e / dof)
    xtx_inv = np.linalg.inv(x.T @ x + 1e-10 * np.eye(k))
    se = np.sqrt(np.clip(np.diag(s2 * xtx_inv), 0.0, None))
    t1 = float(beta[s] / max(se[s], 1e-12))
    t2 = float(beta[s + 1] / max(se[s + 1], 1e-12))
    # F test for pi3 = pi4 = 0 (restrict the last two z columns)
    x_r = x[:, : s + 2]
    beta_r, *_ = np.linalg.lstsq(x_r, d4, rcond=None)
    e_r = d4 - x_r @ beta_r
    ssr_r = float(e_r @ e_r)
    ssr_u = float(e @ e)
    f34 = ((ssr_r - ssr_u) / 2.0) / max(ssr_u / dof, 1e-12)
    return t1, t2, float(max(f34, 0.0))


def hegy_test(
    y: FloatArray,
    s: int = 4,
    mc: int = 800,
    seed: int = 0,
) -> dict[str, float]:
    """HEGY quarterly seasonal unit-root test with MC p-values."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    n = yy.size
    if s != 4:
        raise ValueError("quarterly s=4 only")
    if n < 8 * s:
        raise ValueError("need n >= 8s")
    if float(np.ptp(yy)) <= 1e-12:
        raise ValueError("constant series")
    t1, t2, f34 = _hegy_stats(yy, s)
    rng = np.random.default_rng(seed)
    null1 = np.empty(mc)
    null2 = np.empty(mc)
    nullf = np.empty(mc)
    for b in range(mc):
        # seasonal random walk: all four seasonal roots
        e = rng.normal(size=n)
        ysim = np.empty(n)
        ysim[:s] = e[:s]
        for k in range(s, n):
            ysim[k] = ysim[k - s] + e[k]
        a1, a2, af = _hegy_stats(ysim, s)
        null1[b], null2[b], nullf[b] = a1, a2, af
    # left-tail t-stats, right-tail F
    p1 = float((1.0 + (null1 <= t1).sum()) / (mc + 1))
    p2 = float((1.0 + (null2 <= t2).sum()) / (mc + 1))
    pf = float((1.0 + (nullf >= f34).sum()) / (mc + 1))
    return {
        "t_pi1": t1,
        "t_pi2": t2,
        "f_pi34": f34,
        "p_pi1": p1,
        "p_pi2": p2,
        "p_pi34": pf,
    }


def _ch_stat(y: FloatArray, s: int) -> float:
    """Canova-Hansen L statistic on seasonal dummies."""
    n = y.size
    d = _seasonal_dummies(n, s)
    beta, *_ = np.linalg.lstsq(d, y, rcond=None)
    e = y - d @ beta
    # long-run variance (white-noise form — documented)
    s2 = float(e @ e) / max(n - s, 1)
    if s2 <= 1e-12:
        raise ValueError("zero residual variance")
    cum = np.cumsum(e[:, None] * d, axis=0)
    m_j = (cum * cum).sum(axis=0)
    return float(m_j.sum() / (n * n * s2))


def canova_hansen(
    y: FloatArray,
    s: int = 4,
    mc: int = 800,
    seed: int = 0,
) -> dict[str, float]:
    """Canova-Hansen seasonal stability test (MC p-value)."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    n = yy.size
    if n < 8 * s:
        raise ValueError("need n >= 8s")
    if float(np.ptp(yy)) <= 1e-12:
        raise ValueError("constant series")
    stat = _ch_stat(yy, s)
    rng = np.random.default_rng(seed + 7)
    null = np.empty(mc)
    for b in range(mc):
        ysim = rng.normal(size=n)
        null[b] = _ch_stat(ysim, s)
    p = float((1.0 + (null >= stat).sum()) / (mc + 1))
    return {"ch_stat": stat, "p_value": p}


def bench_hegy(seed: int = 20261231 + 454) -> dict[str, float]:
    """SYNTHETIC check — stable seasonal vs seasonal-RW series."""
    rng = np.random.default_rng(seed)
    n, s = 240, 4
    # stable deterministic seasonality: all roots rejected
    t_idx = np.arange(n)
    y_stable = np.sin(2 * np.pi * t_idx / s) + 0.3 * rng.normal(size=n)
    out_s = hegy_test(y_stable, mc=300, seed=seed)
    # seasonal random walk: all roots present
    e = rng.normal(size=n)
    y_rw = np.empty(n)
    y_rw[:s] = e[:s]
    for k in range(s, n):
        y_rw[k] = y_rw[k - s] + e[k]
    out_rw = hegy_test(y_rw, mc=300, seed=seed + 1)
    ok = out_s["p_pi1"] < 0.10 and out_s["p_pi34"] < 0.10 and out_rw["p_pi34"] > 0.05
    if not ok:
        raise ValueError(
            f"hegy off: stable p34={out_s['p_pi34']:.3f} "
            f"rw p34={out_rw['p_pi34']:.3f} p1={out_s['p_pi1']:.3f}"
        )
    ch_s = canova_hansen(y_stable, mc=300, seed=seed)
    e_rw2 = np.cumsum(rng.normal(size=(n, s)), axis=0).ravel("F")[:n]
    ch_rw = canova_hansen(e_rw2, mc=300, seed=seed + 2)
    if not (ch_s["p_value"] > 0.05 and ch_rw["p_value"] < 0.10):
        raise ValueError(f"ch off: stable p={ch_s['p_value']:.3f} rw p={ch_rw['p_value']:.3f}")
    return {
        "synthetic_stable_p34": out_s["p_pi34"],
        "synthetic_rw_p34": out_rw["p_pi34"],
        "synthetic_ch_stable_p": ch_s["p_value"],
        "synthetic_ch_rw_p": ch_rw["p_value"],
        "synthetic_score": 1.0,
    }
