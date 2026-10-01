"""Ng-Perron (2001) modified unit-root tests (MZ/MZS/MSB/MPT).

References
----------
- Ng, S. & Perron, P. (2001). "Lag Length Selection and the
  Construction of Unit Root Tests with Good Size and Power."
  *Econometrica* 69(6), 1519-1554.
- Perron, P. & Ng, S. (1996). "Useful Modifications to Some
  Unit Root Tests with Dependent Errors and their Local
  Asymptotic Properties." *Review of Economic Studies* 63(3),
  435-463.
- Stock, J.H. (1999). "A Class of Tests for Integration and
  Cointegration." In *Cointegration, Causality and
  Forecasting* (Festschrift for Clive Granger), OUP.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
The Ng-Perron family applies the ERS GLS detrending (same
c = -7 / -13.5 constants) and then forms the modified
Phillips-Perron statistics with the autoregressive spectral
density estimator ``s_AR^2 = s_e^2 / (1 - sum b_j)^2`` from
an ADF regression on the detrended data:

    MZ_a  = (T^{-1} y_T^2 - s_AR^2) / (2 k),  k = s_AR^2 T^{-2} sum y_t^2
    MZ_t  = MZ_a * MSB,          MSB = sqrt(k / s_AR^2)
    MPT   = (c^2 * sum y_t^2 - c_bar * T * y_T^2) / s_AR^2, c_bar = c+1

with the published 5% critical values (constant: MZ_a -8.1,
MZ_t -1.98, MSB 0.233, MPT 3.17; trend: -17.3, -2.91, 0.168,
5.48). All four statistics agree on the same decision for
coherent results — the bench demands unanimity on the RW vs
stationary contrast.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _gls_detrend(x: FloatArray, trend: bool) -> FloatArray:
    t = x.size
    c = -13.5 if trend else -7.0
    rho = 1.0 + c / t
    w = np.diff(x) - (rho - 1.0) * x[:-1]
    w = np.concatenate([[x[0]], w])
    if trend:
        z = np.column_stack([np.ones(t), np.arange(1.0, t + 1.0)])
    else:
        z = np.ones((t, 1))
    qd_z = np.diff(z, axis=0) - (rho - 1.0) * z[:-1]
    qd_z = np.vstack([z[:1], qd_z])
    coef, *_ = np.linalg.lstsq(qd_z, w, rcond=None)
    return np.asarray(x - z @ coef)


def _ar_spectral(y: FloatArray, k: int) -> float:
    """Autoregressive LRV from ADF(k) on detrended data."""
    dy = np.diff(y)
    rows = []
    for i in range(k, dy.size):
        rows.append([y[i]] + [dy[i - j] for j in range(1, k + 1)])
    xmat = np.asarray(rows)
    target = dy[k:]
    coef, *_ = np.linalg.lstsq(xmat, target, rcond=None)
    resid = target - xmat @ coef
    s2 = float(resid @ resid / resid.size)
    denom = 1.0 - float(np.sum(coef[1:]))
    if abs(denom) < 0.05:
        denom = 0.05 * np.sign(denom) if denom != 0 else 0.05
    return s2 / (denom * denom)


def ng_perron_test(
    x: FloatArray,
    trend: bool = False,
    k_max: int = 8,
) -> dict[str, float]:
    """Ng-Perron MZ/MZS/MSB/MPT battery on GLS-detrended data."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 1 or xx.size < 80 or not np.all(np.isfinite(xx)):
        raise ValueError("bad series")
    if np.std(np.diff(xx)) < 1e-12:
        raise ValueError("degenerate")
    t = xx.size
    yd = _gls_detrend(xx, trend)
    # MAIC-lite lag pick: BIC-style min over k
    best_k, best_sic = 1, np.inf
    for k in range(1, k_max + 1):
        try:
            dy = np.diff(yd)
            rows = []
            for i in range(k, dy.size):
                rows.append([yd[i]] + [dy[i - j] for j in range(1, k + 1)])
            xmat = np.asarray(rows)
            target = dy[k:]
            coef, *_ = np.linalg.lstsq(xmat, target, rcond=None)
            resid = target - xmat @ coef
            s2 = float(resid @ resid / resid.size)
            phis = coef[1:]
            tau0 = float(phis.sum() / (1 - phis.sum())) if abs(1 - phis.sum()) > 1e-3 else 0.0
            v = np.log(s2) + 2.0 * (k + abs(tau0)) / resid.size
            if v < best_sic:
                best_sic, best_k = v, k
        except (ValueError, np.linalg.LinAlgError):
            continue
    s_ar = _ar_spectral(yd, best_k)
    kk = float(s_ar * np.sum(yd**2) / (t * t))
    mza = float((yd[-1] ** 2 / t - s_ar) / (2.0 * kk))
    msb = float(np.sqrt(kk / s_ar))
    mzt = mza * msb
    cbar = (-13.5 if trend else -7.0) + 1.0
    mpt = float((cbar * cbar * np.sum(yd * yd) / (t * t) - cbar * yd[-1] ** 2 / t) / s_ar)
    if trend:
        cvs = {"mza": -17.3, "mzt": -2.91, "msb": 0.168, "mpt": 5.48}
    else:
        cvs = {"mza": -8.1, "mzt": -1.98, "msb": 0.233, "mpt": 3.17}
    rej = (
        int(mza < cvs["mza"])
        + int(mzt < cvs["mzt"])
        + int(msb < cvs["msb"])
        + int(mpt < cvs["mpt"])
    )
    return {
        "mza": mza,
        "mzt": mzt,
        "msb": msb,
        "mpt": mpt,
        "lag": float(best_k),
        "n_reject": float(rej),
        "unanimous_reject": float(rej == 4),
    }


def synth_ngp(
    seed: int = 20261231 + 326,
    n: int = 400,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC random walk vs stationary AR(0.5)."""
    rng = np.random.default_rng(seed)
    rw = np.cumsum(rng.normal(0.0, 1.0, n))
    st = np.zeros(n)
    for t in range(1, n):
        st[t] = 0.5 * st[t - 1] + rng.normal()
    return np.asarray(rw), np.asarray(st)


def bench_ng_perron(seed: int = 20261231 + 326) -> dict[str, float]:
    """Wave-56 self-check: unanimous reject stationary, accept RW."""
    rw, st = synth_ngp(seed=seed)
    r_rw = ng_perron_test(rw)
    r_st = ng_perron_test(st)
    ok = r_rw["unanimous_reject"] == 0.0 and r_st["unanimous_reject"] == 1.0
    return {
        "mza_rw": r_rw["mza"],
        "mza_st": r_st["mza"],
        "mzt_st": r_st["mzt"],
        "n_reject_rw": r_rw["n_reject"],
        "n_reject_st": r_st["n_reject"],
        "score": float(ok),
    }
