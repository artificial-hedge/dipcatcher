"""Differential-flatness feedforward + feedback tracking (wave 279).

Double integrator x'' = u is flat with output y = x. A quintic flat-output
trajectory gives the feedforward u_ff = y''''''; adding light feedback tracks
the reference far tighter than pure feedback regulation.
"""

import numpy as np

_SEED = 20261231 + 761


def _ref(t: float, t_end: float = 8.0) -> tuple[float, float, float]:
    # quintic min-jerk from 0 to 1 over [0, t_end]
    s = min(t / t_end, 1.0)
    y = 10 * s**3 - 15 * s**4 + 6 * s**5
    yd = (30 * s**2 - 60 * s**3 + 30 * s**4) / t_end
    ydd = (60 * s - 180 * s**2 + 120 * s**3) / t_end**2
    return y, yd, ydd


def _run(seed: int, feedforward: bool, steps: int = 800) -> float:
    rng = np.random.RandomState(seed)
    x = np.zeros(2)
    errs: list[float] = []
    for t in range(steps):
        y, yd, ydd = _ref(t * 0.01)
        u = (ydd if feedforward else 0.0) + 4.0 * (y - x[0]) + 4.0 * (yd - x[1])
        x = x + 0.01 * np.array([x[1], u]) + 0.00005 * rng.normal(0, 1, 2)
        errs.append(abs(x[0] - y))
    return float(np.mean(errs[100:]))


def bench_flat_track(seed: int = _SEED) -> dict[str, float]:
    err_ff = _run(seed, feedforward=True)
    err_no = _run(seed, feedforward=False)
    return {
        "synthetic_flat_exact": float(err_ff < 0.01),
        "synthetic_flat_better": float(err_ff < 0.7 * err_no),
    }
