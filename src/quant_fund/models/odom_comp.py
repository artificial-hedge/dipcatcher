"""Complementary-filter heading fusion (wave 283).

Yaw rate gyro integrates to a drifting estimate; a noisy compass reads the
true heading. Complementary blend w*gyro_int + (1-w)*compass beats either
channel alone on RMSE.
"""

import numpy as np

_SEED = 20261231 + 787


def _run(seed: int, w: float | None, steps: int = 800) -> float:
    rng = np.random.RandomState(seed)
    truth = 0.0
    gyro_int, comp = 0.0, 0.0
    errs = []
    for t in range(steps):
        rate = np.sin(0.02 * t)
        truth += 0.01 * rate
        gyro_meas = rate + 0.02 * rng.normal() + 0.06 * t / steps  # bias drift
        comp_meas = truth + 0.15 * rng.normal()
        gyro_int += 0.01 * gyro_meas
        if w is None:
            est = gyro_int
        else:
            comp = w * (comp + 0.01 * gyro_meas) + (1 - w) * comp_meas
            est = comp
        errs.append((est - truth) ** 2)
    return float(np.sqrt(np.mean(errs[100:])))


def bench_odom_comp(seed: int = _SEED) -> dict[str, float]:
    rmse_drift = _run(seed, w=None)
    rmse_fuse = _run(seed, w=0.98)
    return {
        "synthetic_odom_fuse": float(rmse_fuse < 0.7 * rmse_drift),
        "synthetic_odom_bound": float(rmse_fuse < 0.5),
    }
