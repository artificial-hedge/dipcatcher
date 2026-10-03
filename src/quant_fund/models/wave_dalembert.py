"""d'Alembert solution of the 1-D wave equation (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def dalembert(f, g, x: np.ndarray, t: float, c: float = 1.0) -> np.ndarray:
    return np.asarray(
        0.5 * (f(x - c * t) + f(x + c * t)) + 0.5 / c * _cumtrapz(g, x - c * t, x + c * t)
    )


def _cumtrapz(g, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise integral of g from a_i to b_i (g sampled on a fine grid)."""
    grid = np.linspace(-20, 20, 40001)
    gv = np.asarray(g(grid))
    gcdf = np.concatenate([[0.0], np.cumsum(0.5 * (gv[1:] + gv[:-1])) * (grid[1] - grid[0])])
    ia = np.clip(((a - grid[0]) / (grid[1] - grid[0])).astype(int), 0, len(gcdf) - 1)
    ib = np.clip(((b - grid[0]) / (grid[1] - grid[0])).astype(int), 0, len(gcdf) - 1)
    return np.asarray(gcdf[ib] - gcdf[ia])


def _bench_wave_dalembert(seed: int = 0) -> float:
    checks = []

    def f(x: np.ndarray) -> np.ndarray:
        return np.asarray(np.exp(-(x**2)))

    def g(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    x = np.linspace(-5, 5, 200)
    # u(x,0) = f
    checks.append(np.allclose(dalembert(f, g, x, 0.0), f(x), atol=1e-9))
    # solution splits into left/right traveling bumps at t=2: peaks at +-2
    u2 = dalembert(f, g, x, 2.0)
    imax = int(np.argmax(u2))
    checks.append(abs(abs(x[imax]) - 2.0) < 0.15)
    # amplitude halves
    checks.append(abs(float(np.max(u2)) - 0.5) < 0.05)
    # wave equation residual u_tt - c^2 u_xx ~ 0 (finite diff on smooth solution)
    t = 0.5
    u = dalembert(f, g, x, t)
    u_tt = (dalembert(f, g, x, t + 1e-3) - 2 * u + dalembert(f, g, x, t - 1e-3)) / 1e-6
    u_xx = np.gradient(np.gradient(u, x), x)
    interior = np.s_[10:-10]
    checks.append(
        float(np.max(np.abs(u_tt - u_xx))) < 1e-1
        or float(np.max(np.abs(u_tt[interior] - u_xx[interior]))) < 0.5
    )
    return float(sum(checks) / len(checks))


def bench_wave_dalembert(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wave_dalembert": _bench_wave_dalembert(seed)}
