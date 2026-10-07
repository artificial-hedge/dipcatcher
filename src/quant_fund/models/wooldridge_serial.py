"""Wooldridge (2002) serial-correlation test for FE panels.

References
----------
- Wooldridge, J.M. (2002). *Econometric Analysis of Cross
  Section and Panel Data*. MIT Press, sec. 10.5.4.
- Drukker, D.M. (2003). "Testing for Serial Correlation in
  Linear Panel-Data Models." *Stata Journal* 3(2), 168-177.
- Arellano, M. & Bond, S. (1991). "Some Tests of
  Specification for Panel Data." *Review of Economic
  Studies* 58(2), 277-297.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
Fixed-effects residuals ``e_it`` carry the unit-removed
idiosyncratic error. Wooldridge's test pools the regression

    e_it = rho * e_{i,t-1} + v_it,  t = 2..T_i,

and evaluates ``rho`` with standard errors clustered by
unit — under the no-serial-correlation null the pooled
slope concentrates at ``-1/(T_i - 1)`` for small T (the
within-transformation bias), and the clustered z on
``rho + 1/(T-1)`` is standard normal; we report the
two-sided decision on the centered coefficient. Related to
the Arellano-Bond m(1)/m(2) tests on first-differenced
errors, which require the panel-GMM context we do not
assume here. The bench generates balanced panels whose
idiosyncratic error follows AR(1) with phi=0.6 (reject)
vs iid (accept).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _cluster_se(x: FloatArray, e: FloatArray, groups: FloatArray) -> float:
    """Clustered variance of the slope in pooled e~x reg."""
    n = x.size
    xx = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(xx, e, rcond=None)
    resid = e - xx @ coef
    uniq = np.unique(groups)
    meat = np.zeros((2, 2))
    for g in uniq:
        idx = groups == g
        sc = (xx[idx].T) @ resid[idx]
        meat += np.outer(sc, sc)
    bread = np.linalg.pinv(xx.T @ xx)
    cov = bread @ meat @ bread
    return float(np.sqrt(max(cov[1, 1], 1e-30)))


def wooldridge_serial(
    resid: FloatArray,
    unit: FloatArray,
) -> dict[str, float]:
    """Wooldridge FE serial-correlation test.

    ``resid`` are within-transformed (unit-demeaned)
    residuals stacked by unit; ``unit`` holds unit ids in
    the same order (sorted contiguous).
    """
    r = np.asarray(resid, dtype=np.float64)
    u = np.asarray(unit)
    if r.ndim != 1 or u.ndim != 1 or r.size != u.size or r.size < 60:
        raise ValueError("bad inputs")
    if not np.all(np.isfinite(r)) or float(np.std(r)) < 1e-12:
        raise ValueError("degenerate")
    lag_e: list[float] = []
    cur_e: list[float] = []
    grp: list[int] = []
    tbar: list[int] = []
    for g in np.unique(u):
        idx = np.where(u == g)[0]
        series = r[idx]
        if series.size < 3:
            continue
        tbar.append(series.size)
        lag_e.extend(series[:-1])
        cur_e.extend(series[1:])
        grp.extend([int(g)] * (series.size - 1))
    x = np.asarray(lag_e, dtype=np.float64)
    e = np.asarray(cur_e, dtype=np.float64)
    g_arr = np.asarray(grp, dtype=np.float64)
    n = x.size
    if n < 40:
        raise ValueError("insufficient lags")
    xx = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(xx, e, rcond=None)
    se = _cluster_se(x, e, g_arr)
    t_bar = float(np.mean(tbar))
    # within-transformation bias benchmark under the null
    center = -1.0 / max(t_bar - 1.0, 1.0)
    z = float((coef[1] - center) / se)
    from scipy.stats import norm

    pval = float(2 * norm.sf(abs(z)))
    return {
        "rho_hat": float(coef[1]),
        "null_center": center,
        "z": z,
        "pval": pval,
        "reject5": float(pval < 0.05),
        "n_pairs": float(n),
        "n_units": float(len(tbar)),
    }


def synth_wooldridge(
    seed: int = 20261231 + 334,
    n_units: int = 60,
    t: int = 8,
    phi: float = 0.6,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC FE residuals: AR(1) panel vs iid panel."""
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units), t).astype(np.float64)
    e_ar = np.zeros(n_units * t)
    e_iid = np.zeros(n_units * t)
    for i in range(n_units):
        shocks = rng.standard_normal(t)
        ar = np.zeros(t)
        for s in range(1, t):
            ar[s] = phi * ar[s - 1] + shocks[s]
        # within-transform: the unit mean removal is what the
        # -1/(T-1) null center corrects for.
        e_ar[i * t : (i + 1) * t] = ar - ar.mean()
        iid = rng.standard_normal(t)
        e_iid[i * t : (i + 1) * t] = iid - iid.mean()
    return e_ar, e_iid, unit


def bench_wooldridge(
    seed: int = 20261231 + 334,
) -> dict[str, float]:
    """Wave-57 self-check: AR(1) idiosyncratic errors reject."""
    e_ar, e_iid, unit = synth_wooldridge(seed=seed)
    r1 = wooldridge_serial(e_ar, unit)
    r0 = wooldridge_serial(e_iid, unit)
    ok = r1["reject5"] == 1.0 and r0["reject5"] == 0.0
    return {
        "synthetic_z_ar": r1["z"],
        "synthetic_pval_ar": r1["pval"],
        "synthetic_z_iid": r0["z"],
        "synthetic_pval_iid": r0["pval"],
        "synthetic_rho_ar": r1["rho_hat"],
        "synthetic_score": float(ok),
    }
