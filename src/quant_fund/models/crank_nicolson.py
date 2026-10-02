"""Crank–Nicolson canon: θ=1/2 trapezoidal rule for the heat
equation u_t = D u_xx on [0,1] with Dirichlet ends — second-order
in time and space, unconditionally stable. Compared against
explicit FTCS which is unstable unless dt ≤ dx²/(2D).
All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def crank_nicolson_heat(
    u0: FloatArray,
    dx: float,
    dt: float,
    steps: int,
    diffusivity: float = 1.0,
) -> FloatArray:
    """Crank–Nicolson time stepping for u_t = D u_xx.

    (I − r/2 δ²) u^{n+1} = (I + r/2 δ²) u^n with r = D·dt/dx².
    Dirichlet ends fixed at 0.
    """
    n = u0.size
    r = diffusivity * dt / dx**2
    # tridiagonal coefficients (interior points only)
    m = n - 2
    diag = np.full(m, 1.0 + r)
    off = np.full(m - 1, -r / 2)
    # build explicit matrices
    Am = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    Bm = (
        np.diag(np.full(m, 1.0 - r))
        + np.diag(np.full(m - 1, r / 2), 1)
        + np.diag(np.full(m - 1, r / 2), -1)
    )
    u = u0.copy()
    for _ in range(steps):
        rhs = Bm @ u[1:-1]
        u[1:-1] = np.linalg.solve(Am, rhs)
    return u


def ftcs_heat(
    u0: FloatArray,
    dx: float,
    dt: float,
    steps: int,
    diffusivity: float = 1.0,
) -> FloatArray:
    """Explicit FTCS u^{n+1} = u^n + r δ²u^n — unstable for r > 1/2."""
    r = diffusivity * dt / dx**2
    u = u0.copy()
    for _ in range(steps):
        u[1:-1] += r * (u[2:] - 2 * u[1:-1] + u[:-2])
    return np.asarray(u, dtype=np.float64)


def heat_exact(x: FloatArray, t: float, diffusivity: float = 1.0, k: float = np.pi) -> FloatArray:
    """Exact decaying mode sin(kx)·exp(−D k² t) on [0,1]."""
    return np.asarray(np.sin(k * x) * np.exp(-diffusivity * k**2 * t), dtype=np.float64)


def bench_crank_nicolson(seed: int = 20261231) -> dict[str, float]:
    """Order-2 time convergence + unconditional stability demo."""
    out: dict[str, float] = {}
    nx = 64
    x = np.linspace(0.0, 1.0, nx + 1)
    dx = x[1] - x[0]
    u0 = np.sin(np.pi * x)
    # time-convergence vs a fine-CN reference on the same grid —
    # isolates the O(dt²) temporal error from the fixed O(dx²)
    # spatial error
    u_fine = crank_nicolson_heat(u0, dx, 0.5 / 3200, 3200)
    errs = {}
    for nt in (200, 400):
        dt = 0.5 / nt
        u = crank_nicolson_heat(u0, dx, dt, nt)
        errs[nt] = float(np.abs(u - u_fine).max())
    out["synthetic_cn_order"] = float(np.log2(errs[200] / errs[400]))
    out["synthetic_cn_err_vs_fine"] = errs[400]
    # also report the true spatial+temporal error at fine resolution
    u_exact_check = crank_nicolson_heat(u0, dx, 0.5 / 1600, 1600)
    out["synthetic_cn_err_exact"] = float(np.abs(u_exact_check - heat_exact(x, 0.5)).max())
    # unconditional stability: r = 8 ≫ 1/2 — FTCS explodes, CN fine
    dt_big = 8.0 * dx**2 / 2.0  # r = 4
    steps = 40
    u_cn = crank_nicolson_heat(u0, dx, dt_big, steps)
    u_ft = ftcs_heat(u0, dx, dt_big, steps)
    out["synthetic_cn_big_dt_finite"] = float(np.isfinite(u_cn).all())
    out["synthetic_cn_big_dt_max"] = float(np.abs(u_cn).max())
    out["synthetic_ftcs_big_dt_max"] = float(np.abs(u_ft).max())
    out["synthetic_stability_ratio"] = out["synthetic_ftcs_big_dt_max"] / max(
        out["synthetic_cn_big_dt_max"], 1e-300
    )
    return out
