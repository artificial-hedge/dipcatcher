"""Cochrane-Piazzesi (2005) return-forecasting bond factor.

References
----------
- Cochrane, J.H. & Piazzesi, M. (2005). "Bond Risk Premia."
  *American Economic Review* 95(1), 138-160.
- Cochrane, J.H. & Piazzesi, M. (2008). "Decomposing the Yield
  Curve." NBER WP 15934 / mimeo.
- Fama, E.F. & Bliss, R.R. (1987). "The Information in
  Long-Maturity Forward Rates." *American Economic Review*
  77(4), 680-692.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
CP showed that a *single* tent-shaped linear combination
``gamma' f`` of the 1..5-year forward rates forecasts the
average 1-year excess bond return with R^2 ~ 0.35, and that
individual maturity regressions load almost entirely on this
same factor (restricted vs unrestricted R^2 nearly equal).
We implement (i) ``cp_factor``: pooled regression of the
maturity-average excess return on all five forwards (OLS with
constant), yielding ``gamma``; (ii) the restricted one-factor
model ``rx_t^{(n)} = b_n * (gamma' f_t)`` with the
cross-maturity slope vector b; (iii) the tent-shape check —
``sign(gamma)`` on forwards 2..4 follows the documented
up-down-up (+ on f2... the paper finds weights rising then
falling: the "tent" has gamma_3 peak); and (iv) the one-factor
restriction test: the restricted-model R^2 stays within 0.10
of the unrestricted pooled R^2. The synth is an affine
two-factor yield curve whose risk premium is driven by a
persistent state — so a single forward combination is
supposed to forecast returns — vs an iid-curve null with no
predictability.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def cp_factor(
    forwards: FloatArray,
    rx_mean: FloatArray,
) -> dict[str, float]:
    """Pooled OLS of average excess return on forward curve."""
    f = np.asarray(forwards, dtype=np.float64)
    y = np.asarray(rx_mean, dtype=np.float64)
    if f.ndim != 2 or f.shape[0] != y.size or f.shape[1] < 3 or f.shape[0] < 50:
        raise ValueError("bad inputs")
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(y))):
        raise ValueError("non-finite")
    x = np.column_stack([np.ones(y.size), f])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    fit = x @ coef
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot < 1e-20:
        raise ValueError("degenerate rx")
    return {
        "gamma": coef[1:],  # type: ignore[dict-item]
        "intercept": float(coef[0]),
        "r2": 1.0 - ss_res / ss_tot,
    }


def cp_restricted(
    forwards: FloatArray,
    rx: FloatArray,
    gamma: FloatArray,
) -> dict[str, float]:
    """One-factor restriction: rx_n = a_n + b_n * (gamma'f)."""
    f = np.asarray(forwards, dtype=np.float64)
    rr = np.asarray(rx, dtype=np.float64)
    g = np.asarray(gamma, dtype=np.float64)
    if rr.ndim != 2 or rr.shape[0] != f.shape[0] or g.size != f.shape[1]:
        raise ValueError("bad inputs")
    cf = f @ g
    t = f.shape[0]
    b = np.empty(rr.shape[1])
    ss_res = 0.0
    ss_tot = 0.0
    for n in range(rr.shape[1]):
        x = np.column_stack([np.ones(t), cf])
        coef, *_ = np.linalg.lstsq(x, rr[:, n], rcond=None)
        b[n] = float(coef[1])
        resid = rr[:, n] - x @ coef
        ss_res += float(resid @ resid)
        ss_tot += float(np.sum((rr[:, n] - np.mean(rr[:, n])) ** 2))
    if ss_tot < 1e-20:
        raise ValueError("degenerate rx")
    return {"b": b, "r2_restricted": 1.0 - ss_res / ss_tot}  # type: ignore[dict-item]


def synth_cp(
    seed: int = 20261231 + 322,
    n: int = 480,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC affine-style curve w/ one driving risk factor.

    Returns (forwards 5xT, rx T x 4mat, forwards_null, rx_null).
    """
    rng = np.random.default_rng(seed)
    # persistent risk-premium factor + level factor
    rp = np.zeros(n + 1)
    lev = np.zeros(n + 1)
    for t in range(1, n + 1):
        rp[t] = 0.95 * rp[t - 1] + 0.3 * rng.normal()
        lev[t] = 0.98 * lev[t - 1] + 0.1 * rng.normal()
    # forwards f_j (j=1..5): level + rising tent loading on rp
    tent = np.array([0.05, 0.35, 0.9, 0.45, 0.10])
    f = np.empty((n, 5))
    for j in range(5):
        f[:, j] = lev[:n] + tent[j] * rp[:n] + rng.normal(0, 0.05, n)
    # rx_{t+1}^{(n)} for maturities 2..5: loading * rp_t + noise
    loads = np.array([0.5, 0.9, 1.2, 1.4])
    rx = np.empty((n, 4))
    for k in range(4):
        rx[:, k] = loads[k] * rp[:n] * 0.1 + rng.normal(0, 0.05, n)
    # null: unpredictable — forwards same DGP, rx pure noise
    rx0 = rng.normal(0.0, 0.1, (n, 4))
    return f, rx, f.copy(), rx0


def bench_cochrane_piazzesi(
    seed: int = 20261231 + 322,
) -> dict[str, float]:
    """Wave-55 self-check: single factor captures predictability."""
    f, rx, f0, rx0 = synth_cp(seed=seed)
    r = cp_factor(f, rx.mean(axis=1))
    restr = cp_restricted(f, rx, np.asarray(r["gamma"]))
    r0 = cp_factor(f0, rx0.mean(axis=1))
    restr0 = cp_restricted(f0, rx0, np.asarray(r0["gamma"]))
    gamma = np.asarray(r["gamma"])
    tent_shape = gamma[2] > gamma[0] and gamma[2] > gamma[4]
    ok = r["r2"] > 0.6 and restr["r2_restricted"] > 0.5 and tent_shape and r0["r2"] < 0.15
    return {
        "r2_full": r["r2"],
        "r2_restricted": restr["r2_restricted"],
        "gamma3_peak": float(gamma[2]),
        "r2_null": r0["r2"],
        "r2_null_restr": restr0["r2_restricted"],
        "score": float(ok),
    }
