"""Sliding-mode control: reaching + sliding phases on double integrator (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 717


def smc_run(x0: np.ndarray, lam: float, k: float, steps: int, dt: float = 0.005) -> np.ndarray:
    """u = -k sign(s) - ...; s = lam*x1 + x2."""
    x = x0.copy()
    hist = []
    for _ in range(steps):
        s = lam * x[0] + x[1]
        u = -k * np.sign(s)
        x = x + dt * np.array([x[1], u])
        hist.append(x.copy())
    return np.array(hist)


def bench_sliding_mode(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 30
    for _ in range(trials):
        x0 = rng.normal(0, 1, 2)
        hist = smc_run(x0, 1.0, 8.0, 600)
        ok += float(np.linalg.norm(hist[-1]) < 0.5)
    return {"synthetic_smc_converges": ok / trials}
