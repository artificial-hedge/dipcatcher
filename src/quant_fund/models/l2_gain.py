"""Closed-loop L2-gain reduction by state feedback (wave 279).

x' = A x + B u + Bd w, y = C x. Feedback u = -K x attenuates the
disturbance-to-output channel: the induced L2 gain (output energy / input
energy) of the closed loop is lower than open loop for the same w.
"""

import numpy as np

_SEED = 20261231 + 763


def _run(seed: int, feedback: bool, steps: int = 1000) -> float:
    rng = np.random.RandomState(seed)
    a_mat = np.array([[0.0, 1.0], [1.0, -0.5]])
    b_mat = np.array([[0.0], [1.0]])
    c_vec = np.array([1.0, 0.0])
    k_gain = np.array([3.0, 3.0])
    x = np.zeros(2)
    ein, eout = 0.0, 0.0
    for _ in range(steps):
        w = float(rng.normal(0, 1))
        u = -float(k_gain @ x) if feedback else 0.0
        x = x + 0.01 * (a_mat @ x + b_mat[:, 0] * (u + w))
        y = float(c_vec @ x)
        ein += w * w
        eout += y * y
    return eout / max(ein, 1e-9)


def bench_l2_gain(seed: int = _SEED) -> dict[str, float]:
    g_open = _run(seed, feedback=False)
    g_closed = _run(seed, feedback=True)
    return {
        "synthetic_l2_attenuate": float(g_closed < 0.7 * g_open),
        "synthetic_l2_stable": float(np.isfinite(g_closed)),
    }
