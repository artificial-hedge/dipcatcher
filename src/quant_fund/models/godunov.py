"""Godunov canon: first-order conservative scheme for inviscid
Burgers u_t + (u²/2)_x = 0 — exact Riemann solver at each
interface (HLL-free, the scalar flux cases) tracking a
Riemann shock at the Rankine–Hugoniot speed; compared with
non-conservative advection which slips the shock speed.
All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def burgers_flux_riemann(ul: float, ur: float) -> float:
    """Exact Godunov flux for f(u)=u²/2 at a single interface."""
    if ul <= ur:  # rarefaction fan
        if ul >= 0:
            return 0.5 * ul * ul
        if ur <= 0:
            return 0.5 * ur * ur
        return 0.0
    # shock: sign of speed s = (ul + ur)/2 picks the state
    s = 0.5 * (ul + ur)
    return 0.5 * ul * ul if s > 0 else 0.5 * ur * ur


def godunov_step(u: FloatArray, dx: float, dt: float) -> FloatArray:
    """One conservative Godunov step (transmissive boundaries)."""
    n = u.size
    fl = np.empty(n + 1)
    ext = np.concatenate(([u[0]], u, [u[-1]]))
    for i in range(n + 1):
        fl[i] = burgers_flux_riemann(ext[i], ext[i + 1])
    un = u - dt / dx * (fl[1:] - fl[:-1])
    return np.asarray(un, dtype=np.float64)


def godunov_burgers(u0: FloatArray, dx: float, dt: float, steps: int) -> FloatArray:
    u = u0.copy()
    for _ in range(steps):
        u = godunov_step(u, dx, dt)
    return np.asarray(u, dtype=np.float64)


def nonconservative_step(u: FloatArray, dx: float, dt: float) -> FloatArray:
    """Naive upwind on u_t + u u_x = 0 (non-conservative) — slips
    the shock speed."""
    un = u.copy()
    pos = u > 0
    un[pos] = u[pos] - dt * u[pos] * (u[pos] - np.roll(u, 1)[pos]) / dx
    un[~pos] = u[~pos] - dt * u[~pos] * (np.roll(u, -1)[~pos] - u[~pos]) / dx
    return np.asarray(un, dtype=np.float64)


def bench_godunov(seed: int = 20261231) -> dict[str, float]:
    """RH shock-speed check + rarefaction fan shape."""
    out: dict[str, float] = {}
    n = 400
    x = np.linspace(0, 2, n, endpoint=False)
    dx = x[1] - x[0]
    ul, ur = 1.0, 0.0
    u0 = np.where(x < 1.0, ul, ur)
    t_final = 0.4
    # CFL ~0.8
    dt = 0.8 * dx / max(abs(ul), abs(ur), 1e-30)
    steps = int(t_final / dt)
    u_g = godunov_burgers(u0, dx, dt, steps)
    s_exact = 0.5 * (ul + ur)
    x_shock = 1.0 + s_exact * t_final
    # numerical shock position: midpoint crossing of 0.5
    below = np.where(u_g < 0.5)[0]
    x_num = x[below[0]] if below.size else np.nan
    out["synthetic_shock_pos_exact"] = float(x_shock)
    out["synthetic_shock_pos_num"] = float(x_num)
    out["synthetic_shock_pos_err"] = float(abs(x_num - x_shock))
    # non-conservative version slips
    u_nc = u0.copy()
    for _ in range(steps):
        u_nc = nonconservative_step(u_nc, dx, dt)
    below_nc = np.where(u_nc < 0.5)[0]
    x_nc = x[below_nc[0]] if below_nc.size else np.nan
    out["synthetic_nc_shock_err"] = float(abs(x_nc - x_shock))
    out["synthetic_conservation_bonus"] = (
        out["synthetic_nc_shock_err"] - out["synthetic_shock_pos_err"]
    )
    # mass: ∫u dx changes only through boundary fluxes
    mass_err = float(abs(u_g.sum() - u0.sum()) * dx - abs(0.5 * ul * ul) * t_final)
    out["synthetic_mass_balance_err"] = abs(mass_err)
    # rarefaction: ul=-1? use ul=0, ur=1 (fan): u(x/t) = x/t
    u0r = np.where(x < 1.0, 0.0, 1.0)
    u_r = godunov_burgers(u0r, dx, dt, steps)
    xx = x - 1.0
    fan = np.clip(xx / t_final, 0.0, 1.0)
    out["synthetic_fan_err"] = float(np.sqrt(((u_r - fan) ** 2).mean()))
    return out
