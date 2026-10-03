"""Lennard-Jones molecular dynamics: velocity-Verlet + LJ force + energy drift."""

import numpy as np

_SEED = 20261231 + 704


def _forces(pos: np.ndarray, eps: float = 1.0, sig: float = 1.0) -> np.ndarray:
    n = len(pos)
    f = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            dr = pos[i] - pos[j]
            r2 = float(dr @ dr) + 1e-12
            r6 = (sig * sig / r2) ** 3
            fmag = 24 * eps * (2 * r6 * r6 - r6) / r2
            f[i] += fmag * dr
            f[j] -= fmag * dr
    return f


def simulate(
    pos: np.ndarray, vel: np.ndarray, steps: int, dt: float = 0.002
) -> tuple[np.ndarray, np.ndarray]:
    f = _forces(pos)
    for _ in range(steps):
        vel = vel + 0.5 * dt * f
        pos = pos + dt * vel
        f = _forces(pos)
        vel = vel + 0.5 * dt * f
    return pos, vel


def _energy(pos: np.ndarray, vel: np.ndarray) -> float:
    ke = 0.5 * (vel * vel).sum()
    n = len(pos)
    pe = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r2 = float(((pos[i] - pos[j]) ** 2).sum())
            r6 = (1.0 / r2) ** 3
            pe += 4 * (r6 * r6 - r6)
    return float(ke + pe)


def bench_lj_md(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 15
    for _ in range(trials):
        pos = rng.rand(6, 2) * 8 + 1
        # enforce min separation 1.3 sigma so no pair starts deep in the repulsive core
        for _try in range(200):
            ok_sep = all(
                np.linalg.norm(pos[i] - pos[j]) > 1.3 for i in range(6) for j in range(i + 1, 6)
            )
            if ok_sep:
                break
            pos = rng.rand(6, 2) * 8 + 1
        vel = rng.normal(0, 0.2, (6, 2))
        e0 = _energy(pos, vel)
        pos2, vel2 = simulate(pos.copy(), vel.copy(), 40, dt=0.001)
        e1 = _energy(pos2, vel2)
        drift = abs(e1 - e0) / (abs(e0) + 1.0)
        ok += float(drift < 0.1)
    return {"synthetic_lj_energy_stable": ok / trials}
