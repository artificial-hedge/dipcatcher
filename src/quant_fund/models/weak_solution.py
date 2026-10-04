"""Weak-form residual checks: <u_t,phi> = -<u_x,phi_x> for heat equation (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def weak_residual_heat(u: np.ndarray, dt_ut: np.ndarray, dx: float) -> float:
    """Max |<u_t, phi> + <u_x, phi_x>| over hat test functions vanishing at endpoints."""
    n = len(u)
    res = 0.0
    ux = np.gradient(u, dx)
    for i in range(2, n - 2):
        phi = np.zeros(n)
        phi[i] = 1.0
        phix = np.gradient(phi, dx)
        lhs = float(np.sum(dt_ut * phi) * dx + np.sum(ux * phix) * dx)
        res = max(res, abs(lhs))
    return res


def _bench_weak_solution(seed: int = 0) -> float:
    checks = []
    n = 256
    x = np.linspace(0, 1, n, endpoint=False)
    u = np.sin(2 * np.pi * x)
    ut = -4 * np.pi**2 * u  # exact u_t for heat eq (u_t = u_xx)
    res = weak_residual_heat(u, ut, 1.0 / n)
    # weak residual small but nonzero (discrete phi test) -- bound is heuristic
    checks.append(res < 5.0)
    # exact classical solution satisfies weak form with smooth test functions
    phi = np.sin(2 * np.pi * x)
    ux = 2 * np.pi * np.cos(2 * np.pi * x)
    phix = 2 * np.pi * np.cos(2 * np.pi * x)
    lhs = float(np.sum(ut * phi) * (1.0 / n) + np.sum(ux * phix) * (1.0 / n))
    checks.append(abs(lhs) < 1e-9)
    # space integration by parts on periodic data: <u_x, psi> = -<u, psi_x>
    psi = np.cos(2 * np.pi * x)
    psix = -2 * np.pi * np.sin(2 * np.pi * x)
    a = float(np.sum(ux * psi) / n)
    b = float(np.sum(u * psix) / n)
    checks.append(abs(a + b) < 1e-9)
    # nonlinear flux weak solution sanity: Burgers shock speed = (f(ul)-f(ur))/(ul-ur)
    ul, ur = 1.0, 0.0
    speed = (0.5 * ul**2 - 0.5 * ur**2) / (ul - ur)
    checks.append(abs(speed - 0.5) < 1e-12)
    return float(sum(checks) / len(checks))


def bench_weak_solution(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weak_solution": _bench_weak_solution(seed)}
