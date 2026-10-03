"""Backstepping controller for strict-feedback chain x1' = x2, x2' = u."""

import numpy as np

_SEED = 20261231 + 720


def backstep(x0: np.ndarray, steps: int, c1: float = 2.0, c2: float = 2.0) -> np.ndarray:
    x = x0.copy()
    dt = 0.005
    hist = []
    for _ in range(steps):
        z1 = x[0]
        alpha = -c1 * z1
        z2 = x[1] - alpha
        u = -z1 - c2 * z2 - c1 * x[1]  # u = -z1 - c2*z2 + d(alpha)/dt, dα/dt = -c1*x2
        x = x + dt * np.array([x[1], u])
        hist.append(np.linalg.norm(x))
    return np.array(hist)


def bench_backstepping(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 30
    for _ in range(trials):
        x0 = rng.normal(0, 2, 2)
        h = backstep(x0, 800)
        ok += float(h[-1] < 0.5 and np.isfinite(h).all())
    return {"synthetic_backstep_converges": ok / trials}
