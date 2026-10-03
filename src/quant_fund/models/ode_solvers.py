"""ODE/SDE solver canon: classical RK4, adaptive Dormand-Prince RK45 with
embedded error control, implicit midpoint, and Euler-Maruyama / Milstein for
Ito SDEs. ``bench_ode_solvers`` gates convergence on y' = -2y (exact
decay), a stiff-ish linear system, and the GBM weak-order moment check.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def rk4_step(f: Callable, t: float, y: FloatArray, h: float) -> FloatArray:
    k1 = np.asarray(f(t, y), dtype=np.float64)
    k2 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k1), dtype=np.float64)
    k3 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k2), dtype=np.float64)
    k4 = np.asarray(f(t + h, y + h * k3), dtype=np.float64)
    return y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_solve(f: Callable, t0: float, t1: float, y0: FloatArray, n: int) -> FloatArray:
    y = np.asarray(y0, dtype=np.float64).copy()
    if n < 1 or t1 <= t0:
        raise ValueError("bad grid")
    h = (t1 - t0) / n
    t = t0
    for _ in range(n):
        y = rk4_step(f, t, y, h)
        t += h
    return y


_DP_B = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0])
_DP_BS = np.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])
_DP_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
_DP_A = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
    ]
)


def rk45_solve(
    f: Callable,
    t0: float,
    t1: float,
    y0: FloatArray,
    tol: float = 1e-8,
    h0: float = 0.05,
    max_steps: int = 20000,
) -> tuple[FloatArray, int]:
    """Dormand-Prince RK45 with local error control; returns (y_end, nsteps)."""
    y = np.asarray(y0, dtype=np.float64).copy()
    if t1 <= t0:
        raise ValueError("bad interval")
    t = t0
    h = min(h0, t1 - t0)
    steps = 0
    d = y.size
    while t < t1 - 1e-15 and steps < max_steps:
        h = min(h, t1 - t)
        k = np.zeros((7, d))
        k[0] = np.asarray(f(t, y), dtype=np.float64)
        for i in range(1, 7):
            yi = y + h * (_DP_A[i - 1, :i] @ k[:i])
            k[i] = np.asarray(f(t + _DP_C[i] * h, yi), dtype=np.float64)
        y5 = y + h * (_DP_B @ k)
        y4 = y + h * (_DP_BS @ k)
        err = float(np.max(np.abs(y5 - y4)) / (tol + 1e-30))
        if err <= 1.0:
            y = y5
            t += h
            steps += 1
        h *= float(np.clip(0.9 * err ** (-0.2), 0.2, 5.0)) if err > 0 else 5.0
    return y, steps


def implicit_midpoint(
    f: Callable, t0: float, t1: float, y0: FloatArray, n: int, newton_it: int = 12
) -> FloatArray:
    """Implicit midpoint via fixed-point iteration on the stage equation."""
    y = np.asarray(y0, dtype=np.float64).copy()
    h = (t1 - t0) / n
    t = t0
    for _ in range(n):
        k = np.asarray(f(t + 0.5 * h, y), dtype=np.float64)
        for _ in range(newton_it):
            k_new = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k), dtype=np.float64)
            if np.max(np.abs(k_new - k)) < 1e-12:
                k = k_new
                break
            k = k_new
        y = y + h * k
        t += h
    return y


def euler_maruyama(
    drift: Callable,
    diff: Callable,
    t0: float,
    t1: float,
    y0: FloatArray,
    n: int,
    rng: np.random.Generator,
) -> FloatArray:
    y = np.asarray(y0, dtype=np.float64).copy()
    h = (t1 - t0) / n
    t = t0
    sq = np.sqrt(h)
    for _ in range(n):
        z = rng.standard_normal(y.size)
        y = y + np.asarray(drift(t, y)) * h + np.asarray(diff(t, y)) * (sq * z)
        t += h
    return y


def milstein(
    drift: Callable,
    diff: Callable,
    ddiff: Callable,
    t0: float,
    t1: float,
    y0: FloatArray,
    n: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Scalar Milstein: adds 0.5*g*g'*(dW^2 - dt)."""
    y = np.asarray(y0, dtype=np.float64).copy()
    if y.size != 1:
        raise ValueError("milstein here is scalar")
    h = (t1 - t0) / n
    t = t0
    sq = np.sqrt(h)
    for _ in range(n):
        z = rng.standard_normal(1)
        g = float(np.asarray(diff(t, y))[0])
        dg = float(np.asarray(ddiff(t, y))[0])
        dw = sq * z[0]
        y = y + np.asarray(drift(t, y)) * h + g * dw + 0.5 * g * dg * (dw * dw - h)
        t += h
    return y


def bench_ode_solvers(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    f = lambda t, y: -2.0 * y  # noqa: E731
    y0 = np.array([1.0])
    exact = np.exp(-2.0)
    out["synthetic_rk4_err"] = float(abs(rk4_solve(f, 0.0, 1.0, y0, 100)[0] - exact))
    y45, steps = rk45_solve(f, 0.0, 1.0, y0, tol=1e-9)
    out["synthetic_rk45_err"] = float(abs(y45[0] - exact))
    out["synthetic_rk45_steps"] = float(steps)
    # Nonlinear: logistic y' = y(1-y); exact y(1) from 0.5.
    fl = lambda t, y: y * (1.0 - y)  # noqa: E731
    yl, _ = rk45_solve(fl, 0.0, 1.0, np.array([0.5]), tol=1e-10)
    out["synthetic_rk45_logistic_err"] = float(abs(yl[0] - 1.0 / (1.0 + np.exp(-1.0))))
    out["synthetic_midpoint_err"] = float(abs(implicit_midpoint(f, 0.0, 1.0, y0, 40)[0] - exact))
    # GBM: dX = mu X dt + sig X dW; E[X(1)] = exp(mu). Weak check on EM + Milstein.
    mu, sig = 0.05, 0.4
    dr = lambda t, y: mu * y  # noqa: E731
    di = lambda t, y: sig * y  # noqa: E731
    ddi = lambda t, y: sig * np.ones_like(y)  # noqa: E731
    ends = np.empty(200)
    for i in range(200):
        r = np.random.default_rng(seed + i + 7000)
        ends[i] = euler_maruyama(dr, di, 0.0, 1.0, np.array([1.0]), 400, r)[0]
    out["synthetic_em_gbm_mean_err"] = float(abs(np.log(ends.mean()) - mu))
    for i in range(200):
        r = np.random.default_rng(seed + i + 7000)
        ends[i] = milstein(dr, di, ddi, 0.0, 1.0, np.array([1.0]), 400, r)[0]
    out["synthetic_mil_gbm_mean_err"] = float(abs(np.log(ends.mean()) - mu))
    return out
