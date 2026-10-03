"""D2Q9 lattice-Boltzmann BGK channel flow; mass conservation."""

import numpy as np

_SEED = 20261231 + 706

_W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
_EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
_EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])


def lbm_run(nx: int, ny: int, steps: int, tau: float = 1.0) -> np.ndarray:
    rho = np.ones((ny, nx))
    u = np.zeros((2, ny, nx))
    f = np.zeros((9, ny, nx))
    for i in range(9):
        f[i] = _W[i] * rho
    for _ in range(steps):
        for i in range(9):
            f[i] = np.roll(f[i], (_EX[i], _EY[i]), axis=(1, 0))
        rho = f.sum(axis=0)
        u[0] = sum(f[i] * _EX[i] for i in range(9)) / rho
        u[1] = sum(f[i] * _EY[i] for i in range(9)) / rho
        u2 = u[0] ** 2 + u[1] ** 2
        for i in range(9):
            cu = 3 * (u[0] * _EX[i] + u[1] * _EY[i])
            feq = _W[i] * rho * (1 + cu + 0.5 * cu * cu - 1.5 * u2)
            f[i] = (1 - 1 / tau) * f[i] + (1 / tau) * feq
    return rho


def bench_lattice_boltzmann(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    del rng  # deterministic sim
    rho = lbm_run(24, 12, 60)
    mass0 = 24 * 12
    mass1 = float(rho.sum())
    ok = float(abs(mass1 - mass0) / mass0 < 1e-6 and np.isfinite(rho).all())
    return {"synthetic_lbm_mass": ok}
