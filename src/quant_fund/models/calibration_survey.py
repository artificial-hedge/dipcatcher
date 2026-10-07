"""Calibration estimation — GREG weights on auxiliary totals.

Deville & Särndal (1992): given sample d_i = 1/pi_i base weights
and auxiliary variables x_i with known population totals X, the
calibrated weights minimize a distance to the base weights
subject to sum_i w_i x_i = X. Under the chi^2 (linear) distance

    w_i = d_i (1 + lambda^T x_i),   lambda solves
    [sum_i d_i x_i x_i^T] lambda = X - sum_i d_i x_i

the result is the generalized regression (GREG) estimator. The
raking/logit distance bounds weights into (lo, hi) by iteratively
clipping (Deville-Särndal-Fox truncated linear).

Honesty: the bench uses a biased PPS sample where the auxiliary
margin is known — GREG must pull the total estimate onto the
truth while base weights miss. Fail-closed on singular auxiliary
matrices or empty margins.

References: Deville & Särndal (1992) "Calibration estimators in
survey sampling", JASA 87:376; Deville, Särndal, Sautory (1993)
"Generalized raking procedures"; Särndal (2007) "The calibration
approach in survey theory and practice".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check(
    y: FloatArray, x: FloatArray, d: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    a = np.asarray(y, dtype=float)
    m = np.asarray(x, dtype=float)
    w = np.asarray(d, dtype=float)
    if a.ndim != 1 or m.ndim != 2 or m.shape[0] != a.size or w.shape != a.shape:
        raise ValueError("bad inputs")
    if (
        a.size < 4
        or not np.isfinite(a).all()
        or not np.isfinite(m).all()
        or not np.isfinite(w).all()
    ):
        raise ValueError("non-finite input")
    if (w <= 0).any():
        raise ValueError("non-positive base weights")
    return a, m, w


def greg_weights(x: FloatArray, d: FloatArray, totals: FloatArray) -> FloatArray:
    """Chi^2-distance calibration weights (linear GREG form).

    ``x`` (n, p) auxiliaries per respondent, ``d`` base weights,
    ``totals`` the known population totals of x.
    """
    m = np.asarray(x, dtype=float)
    w = np.asarray(d, dtype=float)
    t = np.asarray(totals, dtype=float)
    if m.ndim != 2 or t.shape != (m.shape[1],):
        raise ValueError("bad auxiliaries")
    _, _, w = _check(np.zeros(m.shape[0]) + 1.0, m, w)
    a_mat = (m.T * w) @ m
    if np.linalg.matrix_rank(a_mat) < m.shape[1]:
        raise ValueError("singular auxiliary matrix")
    resid = t - (w * m.T).sum(axis=1)
    lam = np.linalg.solve(a_mat, resid)
    out = w * (1.0 + m @ lam)
    return np.asarray(out, dtype=np.float64)


def truncated_calibration(
    x: FloatArray,
    d: FloatArray,
    totals: FloatArray,
    lo: float = 0.2,
    hi: float = 5.0,
    n_iter: int = 30,
) -> FloatArray:
    """Bounded calibration: iterate linear GREG with weight-ratio
    clipping into [lo, hi] * base weights (approximate truncated
    linear distance of Deville-Särndal-Sautory)."""
    if not 0 < lo < 1 < hi:
        raise ValueError("bad bounds")
    m = np.asarray(x, dtype=float)
    w = np.asarray(d, dtype=float)
    t = np.asarray(totals, dtype=float)
    cur = w.copy()
    for _ in range(n_iter):
        try:
            new = greg_weights(m, cur, t)
        except ValueError:
            raise
        ratio = new / cur
        clipped = np.clip(ratio, lo, hi)
        if np.abs(clipped - ratio).max() < 1e-8:
            cur = new
            break
        cur = cur * clipped
    return np.asarray(cur, dtype=np.float64)


def bench_calibration_survey(seed: int = 20261231 + 446) -> dict[str, float]:
    """SYNTHETIC check — GREG recovers the margin-consistent total."""
    rng = np.random.default_rng(seed)
    n_pop = 6000
    x_pop = rng.gamma(2.0, 1.0, n_pop)
    y_pop = 5.0 + 2.0 * x_pop + rng.standard_normal(n_pop)
    pi = np.clip(400 * x_pop / x_pop.sum(), 1e-3, 0.9)
    sel = rng.random(n_pop) < pi
    totals = np.array([x_pop.sum()])
    # multi-draw precision comparison: GREG (calibrated to the
    # true x margin) should cut the HT total's RMSE
    errs_g, errs_b = [], []
    for s2 in range(30):
        r2 = np.random.default_rng(seed + s2 + 1)
        sel2 = r2.random(n_pop) < pi
        x2 = x_pop[sel2][:, None]
        y2 = y_pop[sel2]
        d2 = 1.0 / pi[sel2]
        t = float(y_pop.sum())
        errs_b.append((float((d2 * y2).sum()) - t) / t)
        wg = greg_weights(x2, d2, totals)
        errs_g.append((float((wg * y2).sum()) - t) / t)
    rmse_g = float(np.sqrt(np.mean(np.square(errs_g))))
    rmse_b = float(np.sqrt(np.mean(np.square(errs_b))))
    # margin exactness on the first draw
    wg_last = greg_weights(x_pop[sel][:, None], 1.0 / pi[sel], totals)
    marg_err = abs((wg_last * x_pop[sel]).sum() - totals[0]) / totals[0]
    if rmse_g >= rmse_b or marg_err > 0.01:
        raise ValueError(f"greg off: rmse_g={rmse_g:.4f} rmse_b={rmse_b:.4f} marg={marg_err:.4f}")
    return {
        "synthetic_greg_rmse": rmse_g,
        "synthetic_base_rmse": rmse_b,
        "synthetic_margin_err": marg_err,
        "synthetic_score": 1.0,
    }
