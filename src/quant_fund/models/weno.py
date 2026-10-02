"""WENO canon: fifth-order weighted essentially non-oscillatory
finite-difference flux reconstruction for conservation laws —
Jiang–Shu smoothness indicators picking among three candidate
stencils. Tested on linear advection of a step+Gaussian hybrid
profile: near-shock it stays non-oscillatory while remaining
high-order on smooth parts. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Jiang–Shu ideal weights for left-biased reconstruction at i+1/2
_D = np.array([0.3, 0.6, 0.1])
_EPS = 1e-6


def _weno5_left(vm2: float, vm1: float, v0: float, vp1: float, vp2: float) -> float:
    """WENO5 reconstruct u_{i+1/2}^- from the upwind side."""
    # candidate polynomials
    p0 = (2 * vm2 - 7 * vm1 + 11 * v0) / 6
    p1 = (-vm1 + 5 * v0 + 2 * vp1) / 6
    p2 = (2 * v0 + 5 * vp1 - vp2) / 6
    # smoothness indicators
    b0 = 13 / 12 * (vm2 - 2 * vm1 + v0) ** 2 + (vm2 - 4 * vm1 + 3 * v0) ** 2 / 4
    b1 = 13 / 12 * (vm1 - 2 * v0 + vp1) ** 2 + (vp1 - vm1) ** 2 / 4
    b2 = 13 / 12 * (v0 - 2 * vp1 + vp2) ** 2 + (3 * v0 - 4 * vp1 + vp2) ** 2 / 4
    w = _D / (_EPS + np.array([b0, b1, b2])) ** 2
    w = w / w.sum()
    return float(w[0] * p0 + w[1] * p1 + w[2] * p2)


def weno_advect_step(u: FloatArray, sigma: float) -> FloatArray:
    """One upwind-WENO5 flux step for u_t + a u_x = 0, a>0,
    periodic: du/dt = −(f_{i+1/2} − f_{i−1/2})/dx with f=u
    reconstructed left-biased."""
    n = u.size
    fl = np.empty(n)
    for i in range(n):
        # flux at interface i+1/2: stencil centered on cell i
        vm2 = u[(i - 2) % n]
        vm1 = u[(i - 1) % n]
        v0 = u[i % n]
        vp1 = u[(i + 1) % n]
        vp2 = u[(i + 2) % n]
        fl[i] = _weno5_left(vm2, vm1, v0, vp1, vp2)
    # cell j: du_j = -sigma · (f_{j+1/2} - f_{j-1/2})
    flux_diff = fl - np.roll(fl, 1)
    return np.asarray(u - sigma * flux_diff, dtype=np.float64)


def weno_advect(u0: FloatArray, sigma: float, steps: int) -> FloatArray:
    u = u0.copy()
    for _ in range(steps):
        u = weno_advect_step(u, sigma)
    return np.asarray(u, dtype=np.float64)


def linear_advect(u0: FloatArray, sigma: float, steps: int) -> FloatArray:
    """Plain 5-point centered reconstruction (no WENO switch) —
    oscillates at discontinuities."""
    from quant_fund.models.lax_wendroff import upwind_step

    u = u0.copy()
    for _ in range(steps):
        u = upwind_step(u, sigma)
    return u


def bench_weno(seed: int = 20261231) -> dict[str, float]:
    """Step advection: WENO stays bounded; upwind comparison."""
    out: dict[str, float] = {}
    n = 200
    x = np.linspace(0, 1, n, endpoint=False)
    u0 = ((x > 0.3) & (x < 0.5)).astype(np.float64)
    sigma = 0.5
    steps = int(0.3 / (sigma / n))
    u_w = weno_advect(u0, sigma, steps)
    u_u = linear_advect(u0, sigma, steps)
    # total variation of the numerical solution (spurious
    # oscillations add TV beyond the step's 2·jump)
    tv_w = float(np.abs(np.diff(np.roll(u_w, 1) - u_w)).sum())
    tv_u = float(np.abs(np.diff(np.roll(u_u, 1) - u_u)).sum())
    out["synthetic_weno_tv"] = tv_w
    out["synthetic_upwind_tv"] = tv_u
    out["synthetic_weno_max"] = float(u_w.max())
    out["synthetic_weno_min"] = float(u_w.min())
    # WENO should not overshoot far above 1 or below 0
    out["synthetic_weno_overshoot"] = float(max(u_w.max() - 1.0, 0.0) + max(-u_w.min(), 0.0))
    out["synthetic_upwind_overshoot"] = float(max(u_u.max() - 1.0, 0.0) + max(-u_u.min(), 0.0))
    # smooth region accuracy: advect a sine at high order
    u_s = np.sin(2 * np.pi * x)
    u_ws = weno_advect(u_s, sigma, steps)
    ex = np.sin(2 * np.pi * ((x - sigma * steps / n) % 1.0))
    out["synthetic_weno_smooth_err"] = float(np.sqrt(((u_ws - ex) ** 2).mean()))
    return out
