"""2D Ising Metropolis MCMC: magnetization ordering at low T (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 707


def ising_sweep(s: np.ndarray, beta: float, rng: np.random.RandomState) -> np.ndarray:
    n = s.shape[0]
    for i in range(n):
        for j in range(n):
            nn = s[(i + 1) % n, j] + s[(i - 1) % n, j] + s[i, (j + 1) % n] + s[i, (j - 1) % n]
            dE = 2 * s[i, j] * nn
            if dE <= 0 or rng.rand() < np.exp(-beta * dE):
                s[i, j] *= -1
    return s


def bench_ising_metro(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 15
    for _ in range(trials):
        s = rng.choice([-1, 1], (16, 16))
        for _ in range(15):
            s = ising_sweep(s, 0.8, rng)
        m = abs(float(s.mean()))
        ok += float(m > 0.5)  # low-T ferromagnetic ordering
    return {"synthetic_ising_orders": ok / trials}
