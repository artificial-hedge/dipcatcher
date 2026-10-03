"""Disturbance observer with composite control (wave 279).

Plant: x' = a x + b(u + d) with unknown constant d. Augment the state with
the disturbance estimate; feed back -x - dh to cancel it. The composite
controller removes the steady-state offset a plain controller leaves.
"""

import numpy as np

_SEED = 20261231 + 759


def _run(seed: int, steps: int = 600) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    a, b, d_true, sp = -1.0, 1.0, 0.8, 1.0
    # plain state feedback u = k(sp - x)
    x1 = 0.0
    for _ in range(steps):
        u = 5.0 * (sp - x1)
        x1 += 0.01 * (a * x1 + b * (u + d_true)) + 0.0001 * rng.normal()
    err_plain = abs(sp - x1)
    # composite: observer estimates d; u = k(sp - x) - dh + u_ff
    u_ff = -a * sp / b  # cancels the a x term at setpoint
    x2, dh = 0.0, 0.0
    for _ in range(steps):
        u = 5.0 * (sp - x2) - dh + u_ff
        x2 += 0.01 * (a * x2 + b * (u + d_true)) + 0.0001 * rng.normal()
        resid = a * x2 + b * (u + dh)  # predicted derivative incl. estimate
        actual_rate = a * x2 + b * (u + d_true)
        dh += 0.01 * 2.0 * (actual_rate - resid) * b
    err_comp = abs(sp - x2)
    return err_plain, err_comp


def bench_dist_obsv(seed: int = _SEED) -> dict[str, float]:
    err_plain, err_comp = _run(seed)
    return {
        "synthetic_dist_obsv_reject": float(err_comp < 0.5 * err_plain),
        "synthetic_dist_obsv_track": float(err_comp < 0.05),
    }
