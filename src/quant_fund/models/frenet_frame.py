"""Frenet frame of a helix (wave 287) (SYNTHETIC).

Numerical T,N,B and curvature/torsion of the helix
r(t) = (a cos t, a sin t, b t) match kappa = a/(a^2+b^2),
tau = b/(a^2+b^2).
"""

import numpy as np

_SEED = 20261231 + 808


def _helix(t: float, a: float = 2.0, b: float = 0.7) -> np.ndarray:
    return np.array([a * np.cos(t), a * np.sin(t), b * t])


def _frame(t: float, h: float = 1e-4) -> tuple[float, float]:
    r0, rp, rm = _helix(t), _helix(t + h), _helix(t - h)
    rpp, rmm = _helix(t + 2 * h), _helix(t - 2 * h)
    v = (rp - rm) / (2 * h)
    acc = (rp - 2 * r0 + rm) / h**2
    jerk = (rpp - 2 * rp + 2 * rm - rmm) / (2 * h**3)
    cross = np.cross(v, acc)
    kappa = np.linalg.norm(cross) / np.linalg.norm(v) ** 3
    tau = float((cross @ jerk) / np.linalg.norm(cross) ** 2)
    return float(kappa), float(tau)


def bench_frenet_frame(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    errs = []
    for _ in range(6):
        t = rng.uniform(0.5, 5.0)
        k, tau = _frame(t)
        errs += [abs(k - 2.0 / (4.0 + 0.49)), abs(tau - 0.7 / (4.0 + 0.49))]
    return {"synthetic_frenet": float(max(errs) < 1e-3)}
