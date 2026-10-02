"""Lax–Wendroff canon: second-order explicit scheme for linear
advection u_t + a u_x = 0 — the two-step Taylor expansion
u_j^{n+1} = u_j − σ/2 (u_{j+1} − u_{j-1}) + σ²/2 (u_{j+1} − 2u_j
+ u_{j-1}) with σ = a·dt/dx ≤ 1 (CFL). Compared against first-
order upwind on a smooth wave packet. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def lax_wendroff_step(u: FloatArray, sigma: float) -> FloatArray:
    """One LW step on a periodic grid; sigma = a dt/dx."""
    up = np.roll(u, -1)
    um = np.roll(u, 1)
    return np.asarray(
        u - sigma / 2 * (up - um) + sigma**2 / 2 * (up - 2 * u + um),
        dtype=np.float64,
    )


def upwind_step(u: FloatArray, sigma: float) -> FloatArray:
    """First-order upwind for a > 0: u_j − σ(u_j − u_{j-1})."""
    return np.asarray(u - sigma * (u - np.roll(u, 1)), dtype=np.float64)


def advect(
    u0: FloatArray,
    sigma: float,
    steps: int,
    method: str = "lw",
) -> FloatArray:
    """March a periodic profile by `steps` advection steps."""
    u = u0.copy()
    step = lax_wendroff_step if method == "lw" else upwind_step
    for _ in range(steps):
        u = step(u, sigma)
    return np.asarray(u, dtype=np.float64)


def bench_lax_wendroff(seed: int = 20261231) -> dict[str, float]:
    """Phase/diffusion errors on a Gaussian packet at CFL 0.8."""
    out: dict[str, float] = {}
    n = 200
    x = np.linspace(0, 1, n, endpoint=False)
    u0 = np.exp(-((((x - 0.3) % 1.0 - 0.0) / 0.06) ** 2))
    sigma = 0.8
    # advect for exactly one half-domain trip: a·t = 0.5 → packet
    # lands at x = 0.8
    steps = int(0.5 / (sigma / n))
    u_lw = advect(u0, sigma, steps, "lw")
    u_up = advect(u0, sigma, steps, "up")
    u_exact = np.exp(-((((x - 0.8) % 1.0) / 0.06) ** 2))
    # periodic distance-aware L2 error
    out["synthetic_lw_err"] = float(np.sqrt(((u_lw - u_exact) ** 2).mean()))
    out["synthetic_upwind_err"] = float(np.sqrt(((u_up - u_exact) ** 2).mean()))
    out["synthetic_err_ratio"] = out["synthetic_upwind_err"] / max(out["synthetic_lw_err"], 1e-300)
    # peak damping: upwind diffuses heavily
    out["synthetic_lw_peak"] = float(u_lw.max())
    out["synthetic_upwind_peak"] = float(u_up.max())
    # mass conservation (LW conserves discrete mass)
    out["synthetic_lw_mass_err"] = float(abs(u_lw.sum() - u0.sum()) / max(u0.sum(), 1e-30))
    # CFL violation → LW unstable
    u_bad = advect(u0, 1.5, 40, "lw")
    out["synthetic_cfl_violation_max"] = float(np.abs(u_bad).max())
    return out
