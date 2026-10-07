"""1D particle-in-cell plasma: charge deposit + Poisson solve + push (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 708


def pic_step(
    x: np.ndarray, v: np.ndarray, grid_e: np.ndarray, dx: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    n_g = len(grid_e)
    idx = (x / dx).astype(int) % n_g
    v = v + grid_e[idx] * dt
    x = (x + v * dt) % (n_g * dx)
    return x, v


def solve_field(rho: np.ndarray, dx: float) -> np.ndarray:
    # periodic Poisson via spectral
    n = len(rho)
    rho_k = np.fft.fft(rho)
    k = np.fft.fftfreq(n, dx) * 2 * np.pi
    e_k = np.zeros(n, dtype=complex)
    nz = np.abs(k) > 1e-12
    e_k[nz] = -1j * rho_k[nz] / k[nz]
    return np.real(np.fft.ifft(e_k))


def bench_pic_plasma(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 20
    for _ in range(trials):
        x = rng.rand(64) * 16
        v = rng.normal(0, 0.5, 64)
        rho = np.zeros(32)
        idx = (x / 0.5).astype(int) % 32
        np.add.at(rho, idx, 1.0)
        e = solve_field(rho, 0.5)
        for _ in range(10):
            x, v = pic_step(x, v, e, 0.5, 0.05)
        ok += float(np.isfinite(v).all() and np.abs(v).max() < 100)
    return {"synthetic_pic_bounded": ok / trials}
