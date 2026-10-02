"""ADI canon: Peaceman–Rachford alternating-direction implicit
scheme for 2-D diffusion u_t = D (u_xx + u_yy) — tridiagonal
solves per direction, second-order in space and time,
unconditionally stable. Exact-mode check on sin(πx)sin(πy).
All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _thomas(a: FloatArray, b: FloatArray, c: FloatArray, d: FloatArray) -> FloatArray:
    """Thomas tridiagonal solve (a sub, b diag, c super)."""
    n = b.size
    cp = np.empty(n - 1)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n - 1):
        m = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / m
    dp[n - 1] = (d[n - 1] - a[n - 2] * dp[n - 2]) / (b[n - 1] - a[n - 2] * cp[n - 2])
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return np.asarray(x, dtype=np.float64)


def adi_step(
    u: FloatArray,
    dx: float,
    dt: float,
    diffusivity: float = 1.0,
) -> FloatArray:
    """One Peaceman–Rachford ADI step on an (nx, ny) interior grid
    with zero Dirichlet boundary.

    (I − r/2 δx²) u* = (I + r/2 δy²) u^n
    (I − r/2 δy²) u^{n+1} = (I + r/2 δx²) u*
    """
    r = diffusivity * dt / dx**2
    nx, ny = u.shape
    us = u.copy()
    # x-sweep: for each row solve (I − r/2 δx²)
    a = np.full(nx - 3, -r / 2)
    b = np.full(nx - 2, 1.0 + r)
    c = np.full(nx - 3, -r / 2)
    for j in range(1, ny - 1):
        rhs = u[1:-1, j] + (r / 2) * (u[1:-1, j + 1] - 2 * u[1:-1, j] + u[1:-1, j - 1])
        us[1:-1, j] = _thomas(a, b, c, rhs)
    # y-sweep: for each column solve (I − r/2 δy²)
    un = us.copy()
    a2 = np.full(ny - 3, -r / 2)
    b2 = np.full(ny - 2, 1.0 + r)
    c2 = np.full(ny - 3, -r / 2)
    for i in range(1, nx - 1):
        rhs = us[i, 1:-1] + (r / 2) * (us[i + 1, 1:-1] - 2 * us[i, 1:-1] + us[i - 1, 1:-1])
        un[i, 1:-1] = _thomas(a2, b2, c2, rhs)
    return np.asarray(un, dtype=np.float64)


def adi_heat(
    u0: FloatArray, dx: float, dt: float, steps: int, diffusivity: float = 1.0
) -> FloatArray:
    u = u0.copy()
    for _ in range(steps):
        u = adi_step(u, dx, dt, diffusivity)
    return np.asarray(u, dtype=np.float64)


def bench_adi(seed: int = 20261231) -> dict[str, float]:
    """Exact-mode decay + unconditional-stability check on a 48×48 grid."""
    out: dict[str, float] = {}
    n = 48
    x = np.linspace(0, 1, n + 1)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    u0 = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    dx = x[1] - x[0]
    # exact mode decays as exp(−2π² t)
    t_final = 0.05
    dt = t_final / 100
    u = adi_heat(u0, dx, dt, 100)
    amp_exact = np.exp(-2 * np.pi**2 * t_final)
    # project amplitude
    amp = float((u * u0).sum() / (u0 * u0).sum())
    out["synthetic_adi_amp_err"] = abs(amp - amp_exact)
    out["synthetic_adi_amp_ratio"] = amp / amp_exact
    # big-dt stability: r = 20 ≫ stable — still bounded
    u_big = adi_heat(u0, dx, 20 * dx**2, 10)
    out["synthetic_adi_big_dt_finite"] = float(np.isfinite(u_big).all())
    out["synthetic_adi_big_dt_max"] = float(np.abs(u_big).max())
    return out
