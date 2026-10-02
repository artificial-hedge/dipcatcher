"""BDF canon: backward differentiation formulas BDF1/BDF2 for
stiff ODEs — Newton-solved implicit steps on the stiff linear test
y' = λ(y − cos t) − sin t (exact solution y = cos t). Measured
convergence order vs the theoretical order (BDF1: 1, BDF2: 2).
All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _f(t: float, y: float, lam: float) -> float:
    return float(lam * (y - np.cos(t)) - np.sin(t))


def _dfdy(lam: float) -> float:
    return lam


def bdf1(f, y0: float, t: FloatArray, lam: float = 0.0) -> FloatArray:
    """Implicit Euler (BDF1): y_{n+1} = y_n + h·f(t_{n+1}, y_{n+1}).

    Fixed-point-free for the test problem via one Newton step per
    stage on the linear-in-y residual; for general f falls back to
    secant Newton.
    """
    out = np.empty(len(t))
    out[0] = y0
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        y = out[i - 1]
        tn = t[i]
        # Newton on g(y) = y - y_prev - h f(tn, y)
        g = y - out[i - 1] - h * f(tn, y)
        for _ in range(20):
            # finite-difference Jacobian (scalar)
            dg = 1.0 - h * (f(tn, y + 1e-7) - f(tn, y)) / 1e-7
            step = g / dg
            y -= step
            g = y - out[i - 1] - h * f(tn, y)
            if abs(step) < 1e-13 or abs(g) < 1e-13:
                break
        out[i] = y
    return out


def bdf2(f, y0: float, t: FloatArray) -> FloatArray:
    """BDF2: (3y_{n+1} − 4y_n + y_{n-1})/(2h) = f(t_{n+1}, y_{n+1})."""
    out = np.empty(len(t))
    out[0] = y0
    # bootstrap with one implicit-Euler step
    h = t[1] - t[0]
    y = y0
    g = y - y0 - h * f(t[1], y)
    for _ in range(20):
        dg = 1.0 - h * (f(t[1], y + 1e-7) - f(t[1], y)) / 1e-7
        step = g / dg
        y -= step
        g = y - y0 - h * f(t[1], y)
        if abs(step) < 1e-13:
            break
    out[1] = y
    for i in range(2, len(t)):
        h = t[i] - t[i - 1]
        tn = t[i]
        rhs = (4.0 * out[i - 1] - out[i - 2]) / 3.0
        y = out[i - 1]
        # Newton on g = y - rhs - (2h/3) f(tn, y)
        g = y - rhs - (2.0 * h / 3.0) * f(tn, y)
        for _ in range(20):
            dg = 1.0 - (2.0 * h / 3.0) * (f(tn, y + 1e-7) - f(tn, y)) / 1e-7
            step = g / dg
            y -= step
            g = y - rhs - (2.0 * h / 3.0) * f(tn, y)
            if abs(step) < 1e-13:
                break
        out[i] = y
    return out


def _measure_order(err_h: float, err_h2: float) -> float:
    return float(np.log2(err_h / err_h2))


def bench_bdf(seed: int = 20261231) -> dict[str, float]:
    """Stiff-test convergence orders: BDF1→1, BDF2→2, plus a stiff
    stability demonstration (λ=-500, h=0.05 stable where explicit
    Euler explodes)."""
    out: dict[str, float] = {}
    lam = -15.0
    f = lambda t, y: _f(t, y, lam)  # noqa: E731
    t1 = np.linspace(0.0, 2.0, 201)
    t2 = np.linspace(0.0, 2.0, 401)
    exact1 = np.cos(t1)
    exact2 = np.cos(t2)
    e1 = abs(bdf1(f, 1.0, t1)[-1] - exact1[-1])
    e2 = abs(bdf1(f, 1.0, t2)[-1] - exact2[-1])
    out["synthetic_bdf1_order"] = _measure_order(e1, e2)
    g1 = abs(bdf2(f, 1.0, t1)[-1] - exact1[-1])
    g2 = abs(bdf2(f, 1.0, t2)[-1] - exact2[-1])
    out["synthetic_bdf2_order"] = _measure_order(g1, g2)
    # stiff stability: λ=-500, h=0.05 → hλ=-25, outside explicit
    # Euler's stability region [-2,0] but fine for BDF
    lam_stiff = -500.0
    fs = lambda t, y: _f(t, y, lam_stiff)  # noqa: E731
    ts = np.linspace(0.0, 1.0, 21)  # h = 0.05
    yb = bdf1(fs, 1.0, ts)
    out["synthetic_bdf1_stiff_err"] = float(abs(yb[-1] - np.cos(1.0)))
    # explicit Euler on the same grid explodes
    ye = np.empty(len(ts))
    ye[0] = 1.0
    for i in range(1, len(ts)):
        ye[i] = ye[i - 1] + 0.05 * fs(ts[i - 1], ye[i - 1])
    out["synthetic_euler_stiff_err"] = float(abs(ye[-1] - np.cos(1.0)))
    out["synthetic_stiff_err_ratio"] = out["synthetic_euler_stiff_err"] / max(
        out["synthetic_bdf1_stiff_err"], 1e-300
    )
    return out
