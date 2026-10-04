"""Energy method for the heat equation: dE/dt = -||grad u||^2 <= 0 (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def energy(u: np.ndarray, dx: float) -> float:
    return float(np.trapezoid(u**2, dx=dx))


def grad_energy(u: np.ndarray, dx: float) -> float:
    du = np.gradient(u, dx)
    return float(np.trapezoid(du**2, dx=dx))


def heat_step(u: np.ndarray, dt: float, dx: float) -> np.ndarray:
    """Explicit u_t = u_xx step on a periodic grid."""
    return np.asarray(u + dt * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2)


def _bench_energy_method(seed: int = 0) -> float:
    checks = []
    n = 256
    x = np.linspace(0, 1, n, endpoint=False)
    u = np.sin(2 * np.pi * x) + 0.3 * np.sin(10 * np.pi * x)
    e0 = energy(u, 1.0 / n)
    g0 = grad_energy(u, 1.0 / n)
    checks.append(e0 > 0 and g0 > 0)
    # evolve: energy decays monotonically
    dt = 0.1 / n**2
    last = e0
    monotone = True
    for _ in range(500):
        u = heat_step(u, dt, 1.0 / n)
        e = energy(u, 1.0 / n)
        if e > last + 1e-10:
            monotone = False
        last = e
    checks.append(monotone)
    # rate: single mode sin(2pi x) decays at rate 2*(2pi)^2
    v = np.sin(2 * np.pi * x)
    e1 = energy(v, 1.0 / n)
    for _ in range(2000):
        v = heat_step(v, dt, 1.0 / n)
    e2 = energy(v, 1.0 / n)
    t_total = 2000 * dt
    rate = float(-np.log(e2 / e1) / (2 * t_total))
    checks.append(abs(rate - 4 * np.pi**2) / (4 * np.pi**2) < 0.05)
    # energy of zero initial data stays zero
    checks.append(energy(np.zeros(n), 1.0 / n) == 0.0)
    return float(sum(checks) / len(checks))


def bench_energy_method(seed: int = 0) -> dict[str, float]:
    return {"synthetic_energy_method": _bench_energy_method(seed)}
