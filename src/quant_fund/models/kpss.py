"""KPSS (Kwiatkowski-Phillips-Schmidt-Shin 1992) stationarity test.

References
----------
- Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. & Shin, Y.
  (1992). "Testing the Null Hypothesis of Stationarity
  Against the Alternative of a Unit Root." *Journal of
  Econometrics* 54(1-3), 159-178.
- Hobijn, B., Franses, P.H. & Ooms, M. (2004).
  "Generalizations of the KPSS-Test for Stationarity."
  *Statistica Neerlandica* 58(4), 483-502.
- Newey, W.K. & West, K.D. (1994). "Automatic Lag Bandwidth
  Selection." *Econometric Theory* 10(3), 631-653.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
KPSS inverts the usual testing logic: H0 is (level- or
trend-) stationarity, and the statistic is the normalized
partial-sum variance ratio

    eta = (1/T^2) * sum_t S_t^2 / s^2(l),

where S_t is the cumulated residual from the level (or
trend) regression and s^2(l) the Newey-West long-run
variance estimator with Bartlett weights. We use the
canonical Schwert-free bandwidth ``l = trunc(4 (T/100)^0.25)``
(KPSS's own l4/l8/l12 ladder, defaulting to l4) and the
published asymptotic critical values for the constant-mean
(level) and trend cases (5%: 0.463 / 0.146; 1%: 0.739 /
0.216). The unit-root suite complement: DF-GLS and Ng-Perron
test *for* a unit root, KPSS confirms the null of
stationarity — the bench requires the joint decision to be
coherent (random walk: KPSS rejects, AR(0.5): accepts).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# KPSS (1992) Table 1 asymptotic critical values
_CV_LEVEL = {0.10: 0.347, 0.05: 0.463, 0.025: 0.574, 0.01: 0.739}
_CV_TREND = {0.10: 0.119, 0.05: 0.146, 0.025: 0.176, 0.01: 0.216}


def _lrv(u: FloatArray, lag: int) -> float:
    """Newey-West long-run variance, Bartlett window."""
    t = u.size
    v = float(u @ u / t)
    for k in range(1, lag + 1):
        g = float(u[k:] @ u[:-k] / t)
        v += 2.0 * (1.0 - k / (lag + 1)) * g
    return v


def kpss_test(
    x: FloatArray,
    trend: bool = False,
    lags: int | None = None,
) -> dict[str, float]:
    """KPSS level/trend-stationarity test."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 1 or xx.size < 50 or not np.all(np.isfinite(xx)):
        raise ValueError("bad series")
    t = xx.size
    if trend:
        z = np.column_stack([np.ones(t), np.arange(1.0, t + 1.0)])
    else:
        z = np.ones((t, 1))
    coef, *_ = np.linalg.lstsq(z, xx, rcond=None)
    u = xx - z @ coef
    if float(np.std(u)) < 1e-10:
        raise ValueError("degenerate series")
    lag = lags if lags is not None else int(4 * (t / 100) ** 0.25)
    lag = max(lag, 1)
    s = np.cumsum(u)
    lrv = _lrv(u, lag)
    if lrv <= 0:
        raise ValueError("degenerate long-run variance")
    eta = float(s @ s / (t * t * lrv))
    cvs = _CV_TREND if trend else _CV_LEVEL
    p = (
        0.01
        if eta >= cvs[0.01]
        else (0.025 if eta >= cvs[0.025] else (0.05 if eta >= cvs[0.05] else 0.10))
    )
    return {
        "eta": eta,
        "lags": float(lag),
        "cv5": float(cvs[0.05]),
        "p": float(p),
        "reject_stationarity": float(eta >= cvs[0.05]),
    }


def synth_kpss(
    seed: int = 20261231 + 324,
    n: int = 500,
    rho: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC random walk (nonstationary) vs AR(0.5)."""
    rng = np.random.default_rng(seed)
    rw = np.cumsum(rng.normal(0.0, 1.0, n))
    st = np.zeros(n)
    for t in range(1, n):
        st[t] = 0.5 * st[t - 1] + rng.normal()
    return np.asarray(rw), np.asarray(st)


def bench_kpss(seed: int = 20261231 + 324) -> dict[str, float]:
    """Wave-56 self-check: RW rejected, AR(0.5) accepted."""
    rw, st = synth_kpss(seed=seed)
    r_rw = kpss_test(rw)
    r_st = kpss_test(st)
    ok = r_rw["reject_stationarity"] == 1.0 and r_st["reject_stationarity"] == 0.0
    return {
        "synthetic_eta_rw": r_rw["eta"],
        "synthetic_eta_st": r_st["eta"],
        "synthetic_cv5": r_st["cv5"],
        "synthetic_score": float(ok),
    }
