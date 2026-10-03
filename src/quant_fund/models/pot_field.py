"""Artificial-potential-field navigation (wave 283).

Attractive gradient to goal + repulsive 1/d^2 obstacles; the composite field
reaches the goal while a purely attractive controller walks into obstacles.
"""

import numpy as np

_SEED = 20261231 + 785


def _run(seed: int, repulse: bool, steps: int = 800) -> float:
    rng = np.random.RandomState(seed)
    pos = np.array([0.5, 0.5])
    goal = np.array([9.0, 9.0])
    obs = np.array([[4.5, 4.5], [6.0, 5.5], [7.0, 7.0]])
    for _ in range(steps):
        g = (goal - pos) / max(np.linalg.norm(goal - pos), 1e-6)
        u = g.copy()
        if repulse:
            for o in obs:
                d = pos - o
                dist = np.linalg.norm(d)
                if dist < 2.0:
                    u += 0.8 * d / max(dist**3, 1e-3)
        un = u / max(np.linalg.norm(u), 1e-6)
        pos = pos + 0.05 * un + 0.001 * rng.normal(0, 1, 2)
        if np.linalg.norm(pos - goal) < 0.3:
            return 0.0
        if min(np.linalg.norm(pos - o) for o in obs) < 0.25:
            return float(np.linalg.norm(pos - goal))
    return float(np.linalg.norm(pos - goal))


def bench_pot_field(seed: int = _SEED) -> dict[str, float]:
    err_rep = _run(seed, repulse=True)
    err_att = _run(seed, repulse=False)
    return {
        "synthetic_pot_reach": float(err_rep < 0.5),
        "synthetic_pot_better": float(err_rep < err_att),
    }
