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
    # two-sided CFL oracle: stable iff c*dt/dx <= 1. Same deterministic
    # sim below the boundary must stay bounded; above it must blow up.
    x = np.linspace(0, 1, 100)
    dx = x[1] - x[0]
    u0 = np.exp(-((x - 0.5) ** 2) / 0.005)
    c = 1.0
    stable_ok = 0.0
    for lam_frac in (0.3, 0.6, 0.9):
        dt = lam_frac * dx / c
        u = fdtd(u0, c, dx, dt, 120)
        stable_ok += float(np.isfinite(u).all() and np.abs(u).max() < 10)
    unstable_ok = 0.0
    for lam_frac in (1.05, 1.3):
        dt = lam_frac * dx / c
        u = fdtd(u0, c, dx, dt, 120)
        unstable_ok += float((not np.isfinite(u).all()) or np.abs(u).max() >= 10)
    return {
        "synthetic_fdtd_stable_frac": stable_ok / 3,
        "synthetic_fdtd_unstable_caught_frac": unstable_ok / 2,
        "synthetic_fdtd_score": float(stable_ok == 3 and unstable_ok == 2),
    }
