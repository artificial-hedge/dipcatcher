"""Shared fixture for wave-193 optimal-control canon — double-integrator (SYNTHETIC)
plant (position+velocity, force input) with quadratic stage cost;
reference-tracking eval vs a PD-controller baseline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

DT = 0.1
HORIZON = 40


def dyn(x: FloatArray, u: float) -> FloatArray:
    """Double integrator: p' = p + dt*v, v' = v + dt*u."""
    return np.array([x[0] + DT * x[1], x[1] + DT * u])


def dyn_jac(x: FloatArray, u: float) -> tuple[FloatArray, FloatArray]:
    return np.array([[1.0, DT], [0.0, 1.0]]), np.array([0.0, DT])


def stage_cost(x: FloatArray, u: float, target: float = 1.0) -> float:
    return float((x[0] - target) ** 2 + 0.01 * x[1] ** 2 + 0.001 * u**2)


def rollout(u_seq: FloatArray, x0: FloatArray, target: float = 1.0) -> tuple[FloatArray, float]:
    x = np.array(x0, dtype=np.float64)
    tot = 0.0
    xs = [x.copy()]
    for u in u_seq:
        x = dyn(x, float(u))
        tot += stage_cost(x, float(u), target)
        xs.append(x.copy())
    return np.asarray(xs), tot


def pd_baseline(
    x0: FloatArray, target: float = 1.0, kp: float = 0.6, kd: float = 1.2
) -> tuple[FloatArray, float]:
    x = np.array(x0, dtype=np.float64)
    tot = 0.0
    xs = [x.copy()]
    for _ in range(HORIZON):
        u = kp * (target - x[0]) - kd * x[1]
        x = dyn(x, u)
        tot += stage_cost(x, u, target)
        xs.append(x.copy())
    return np.asarray(xs), tot
