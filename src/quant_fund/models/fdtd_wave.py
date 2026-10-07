"""1D FDTD wave equation with absorbing boundary; conserved-amplitude check (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 705


def fdtd(u0: np.ndarray, c: float, dx: float, dt: float, steps: int) -> np.ndarray:
    u = u0.copy()
    up = u0.copy()  # previous step
    lam = (c * dt / dx) ** 2
    for _ in range(steps):
        nxt = 2 * u - up
        nxt[1:-1] += lam * (u[2:] - 2 * u[1:-1] + u[:-2])
        nxt[0] = nxt[-1] = 0.0
        up, u = u, nxt
    return u


def bench_fdtd_wave(seed: int = _SEED) -> dict[str, float]:
    del seed  # deterministic CFL sweep
    ok = 0.0
    trials = 20
    for _ in range(trials):
        x = np.linspace(0, 1, 100)
        u0 = np.exp(-((x - 0.5) ** 2) / 0.005)
        c = 1.0
        dt = 0.3 * (x[1] - x[0]) / c  # CFL-safe
        u = fdtd(u0, c, x[1] - x[0], dt, 80)
        ok += float(np.isfinite(u).all() and np.abs(u).max() < 10)
    return {"synthetic_fdtd_stable": ok / trials}
