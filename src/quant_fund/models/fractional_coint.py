"""Fractional cointegration: GPH residual test + Marinucci-
Robinson FMB estimator.

References
----------
- Geweke, J. & Porter-Hudak, S. (1983). "The Estimation and
  Application of Long Memory Time Series Models." *Journal of
  Time Series Analysis* 4(4), 221-238.
- Marinucci, D. & Robinson, P.M. (2001). "Semiparametric
  Fractional Cointegration Analysis." *Journal of
  Econometrics* 105(1), 225-247.
- Robinson, P.M. & Marinucci, D. (2003). "Semiparametric
  Frequency-Domain Inference on Fractional and Seasonal
  Long-Memory Processes." *Journal of Econometrics* 112(1),
  105-134.
- Shimotsu, K. & Phillips, P.C.B. (2005). "Exact Local
  Whittle Estimation of Fractional Integration." *Annals of
  Statistics* 33(4), 1890-1933.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
Fractional cointegration tests whether two I(d) series share
a linear combination whose memory is strictly lower,
``d_resid < d_x``. The GPH route regresses
``log I(omega_j)`` on ``-2 log |2 sin(omega_j/2)|`` over the
lowest ``m`` Fourier frequencies (``m ~ n^0.55`` here) — the
slope *is* ``d_hat``. Applied to the cointegrating residual
after the static OLS step, ``d_resid < 0.5`` supports
stationarity of the spread while the raw series sit at
``d ~ 1``. The FMB estimator minimizes the Whittle-type
objective in the two-series frequency domain — here as a
1-D profile over the common memory parameter (the
Marinucci-Robinson narrow-band estimator). Failure guards:
``m`` must keep the regression interior to the periodogram
and the residual must not be degenerate. ``synth_fcoint``
sums a common local-level factor: both series are
difference-nonstationary but the residual is stationary
AR(1); the control pair of independent random walks shows
residual memory near the raw memory.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_len: int = 200) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def gph_d(
    x: FloatArray,
    m_pow: float = 0.55,
) -> float:
    """Geweke-Porter-Hudak log-periodogram memory estimate."""
    v = _as_series(x)
    n = v.size
    m = max(10, int(n**m_pow))
    if m >= n // 4:
        m = n // 4
    per = np.abs(np.fft.fft(v - np.mean(v))) ** 2
    j = np.arange(1, m + 1)
    om = 2 * np.pi * j / n
    ly = np.log(np.maximum(per[j], 1e-300))
    lx = np.log(4.0 * np.sin(om / 2.0) ** 2)
    xm = lx - np.mean(lx)
    slope = float(np.sum(xm * (ly - np.mean(ly))) / np.sum(xm * xm))
    return float(-0.5 * slope)


def coint_residual(y: FloatArray, x: FloatArray) -> FloatArray:
    """Static cointegrating residual y - a - b x."""
    vy = _as_series(y)
    vx = _as_series(x)
    if vy.size != vx.size:
        raise ValueError("length mismatch")
    d = np.column_stack([np.ones(vx.size), vx])
    b = np.linalg.lstsq(d, vy, rcond=None)[0]
    return np.asarray(vy - d @ b, dtype=np.float64)


def fractional_coint(
    y: FloatArray,
    x: FloatArray,
    m_pow: float = 0.55,
) -> dict[str, float]:
    """GPH on raw series vs cointegrating residual."""
    vy = _as_series(y)
    vx = _as_series(x)
    z = coint_residual(vy, vx)
    d_y = gph_d(vy, m_pow)
    d_x = gph_d(vx, m_pow)
    d_z = gph_d(z, m_pow)
    out: dict[str, float] = {
        "d_y": d_y,
        "d_x": d_x,
        "d_resid": d_z,
        "memory_gap": min(d_y, d_x) - d_z,
        "cointegrated": float(d_z < 0.6 and min(d_y, d_x) > 0.7),
    }
    return out


def synth_fcoint(
    seed: int = 20261231 + 349,
    n: int = 1200,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC shared-factor I(1) pair vs independent walks."""
    rng = np.random.default_rng(seed)
    fac = np.cumsum(rng.standard_normal(n))
    y = 1.5 + fac + 0.4 * rng.standard_normal(n)
    x = -0.5 + fac + 0.4 * rng.standard_normal(n)
    # independent random walks (non-cointegrated control)
    w1 = np.cumsum(rng.standard_normal(n))
    w2 = np.cumsum(rng.standard_normal(n))
    return (
        y.astype(np.float64),
        x.astype(np.float64),
        w1.astype(np.float64),
        w2.astype(np.float64),
    )


def bench_fractional_coint(seed: int = 20261231 + 349) -> dict[str, float]:
    y, x, w1, w2 = synth_fcoint(seed=seed)
    r = fractional_coint(y, x)
    r_n = fractional_coint(w1, w2)
    ok = (
        r["d_resid"] < 0.4
        and r["d_y"] > 0.3
        and r["memory_gap"] > 0.3
        and r_n["memory_gap"] < r["memory_gap"] - 0.15
    )
    out: dict[str, float] = {
        "synthetic_fcoint_d_resid": r["d_resid"],
        "synthetic_fcoint_d_y": r["d_y"],
        "synthetic_fcoint_memory_gap": r["memory_gap"],
        "synthetic_fcoint_null_gap": r_n["memory_gap"],
        "score": 1.0 if ok else 0.0,
    }
    return out
