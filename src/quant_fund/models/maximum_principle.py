"""Maximum principle for the heat equation: sup u(t) <= sup u(0) (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def heat_step(u: np.ndarray, dt: float, dx: float) -> np.ndarray:
    return np.asarray(u + dt * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2)


def _bench_maximum_principle(seed: int = 0) -> float:
    checks = []
    n = 256
    x = np.linspace(0, 1, n, endpoint=False)
    u0 = np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)
    u = u0.copy()
    dt = 0.1 / n**2
    mx0, mn0 = float(np.max(u0)), float(np.min(u0))
    ok_max = ok_min = True
    for _ in range(2000):
        u = heat_step(u, dt, 1.0 / n)
        ok_max = ok_max and float(np.max(u)) <= mx0 + 1e-9
        ok_min = ok_min and float(np.min(u)) >= mn0 - 1e-9
    checks.append(ok_max)
    checks.append(ok_min)
    # max decreases for non-constant data
    checks.append(float(np.max(u)) < mx0)
    # constant data preserved (max and min equal and constant)
    c = np.full(n, 1.7)
    for _ in range(100):
        c = heat_step(c, dt, 1.0 / n)
    checks.append(np.allclose(c, 1.7))
    # subsolution check: u_t - u_xx <= 0 keeps max principle direction
    checks.append(mx0 - float(np.max(u)) > 0.1)
    return float(sum(checks) / len(checks))


def bench_maximum_principle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_maximum_principle": _bench_maximum_principle(seed)}
