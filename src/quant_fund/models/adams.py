"""Adams canon: Adams–Bashforth explicit multisteps (AB2, AB3)
and the AB2/AM3 predictor–corrector pair, on the scalar test
y' = λ(y − cos t) − sin t (exact y = cos t). Measured orders
vs theory (AB2: 2, AB3: 3, PC: 3). All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _f(t: float, y: float, lam: float) -> float:
    return float(lam * (y - np.cos(t)) - np.sin(t))


def _rk4_bootstrap(f, y0: float, t0: float, h: float, steps: int) -> FloatArray:
    """Classic RK4 for multistep startup values."""
    ys = np.empty(steps + 1)
    ys[0] = y0
    t = t0
    y = y0
    for i in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h * k1 / 2)
        k3 = f(t + h / 2, y + h * k2 / 2)
        k4 = f(t + h, y + h * k3)
        y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h
        ys[i + 1] = y
    return ys


def ab2(f, y0: float, t: FloatArray) -> FloatArray:
    """Adams–Bashforth 2: y_{n+1} = y_n + h(3f_n − f_{n-1})/2."""
    out = np.empty(len(t))
    out[0] = y0
    h = t[1] - t[0]
    out[1] = _rk4_bootstrap(f, y0, t[0], h, 1)[1]
    f_prev = f(t[0], out[0])
    for i in range(1, len(t) - 1):
        f_cur = f(t[i], out[i])
        out[i + 1] = out[i] + h * (1.5 * f_cur - 0.5 * f_prev)
        f_prev = f_cur
    return out


def ab3(f, y0: float, t: FloatArray) -> FloatArray:
    """AB3: y_{n+1} = y_n + h(23f_n − 16f_{n-1} + 5f_{n-2})/12."""
    out = np.empty(len(t))
    out[0] = y0
    h = t[1] - t[0]
    boot = _rk4_bootstrap(f, y0, t[0], h, 2)
    out[1], out[2] = boot[1], boot[2]
    f2, f1 = f(t[0], out[0]), f(t[1], out[1])
    for i in range(2, len(t) - 1):
        f0 = f(t[i], out[i])
        out[i + 1] = out[i] + h * (23 * f0 - 16 * f1 + 5 * f2) / 12
        f2, f1 = f1, f0
    return out


def abm3(f, y0: float, t: FloatArray) -> FloatArray:
    """AB2 predict / Adams–Moulton-3 (trapezoid AM2) correct — a
    classic PECE pair, order 3 overall with two corrector passes."""
    out = np.empty(len(t))
    out[0] = y0
    h = t[1] - t[0]
    boot = _rk4_bootstrap(f, y0, t[0], h, 2)
    out[1], out[2] = boot[1], boot[2]
    f2, f1 = f(t[0], out[0]), f(t[1], out[1])
    for i in range(2, len(t) - 1):
        f0 = f(t[i], out[i])
        # AB3 predictor
        y_p = out[i] + h * (23 * f0 - 16 * f1 + 5 * f2) / 12
        # AM3 corrector (5 f_{n+1} + 8 f_n − f_{n-1})/12
        fp = f(t[i + 1], y_p)
        y_c = out[i] + h * (5 * fp + 8 * f0 - f1) / 12
        fc = f(t[i + 1], y_c)
        out[i + 1] = out[i] + h * (5 * fc + 8 * f0 - f1) / 12
        f2, f1 = f1, f0
    return out


def bench_adams(seed: int = 20261231) -> dict[str, float]:
    """Convergence orders on the stiff-lite test at λ=-10."""
    out: dict[str, float] = {}
    lam = -10.0
    f = lambda t, y: _f(t, y, lam)  # noqa: E731
    errs = {}
    for n in (200, 400):
        t = np.linspace(0.0, 2.0, n + 1)
        exact = np.cos(t)
        errs[n] = {
            "ab2": abs(ab2(f, 1.0, t)[-1] - exact[-1]),
            "ab3": abs(ab3(f, 1.0, t)[-1] - exact[-1]),
            "abm3": abs(abm3(f, 1.0, t)[-1] - exact[-1]),
        }
    for k in ("ab2", "ab3", "abm3"):
        out[f"synthetic_{k}_order"] = float(np.log2(errs[200][k] / errs[400][k]))
        out[f"synthetic_{k}_err"] = float(errs[400][k])
    return out
