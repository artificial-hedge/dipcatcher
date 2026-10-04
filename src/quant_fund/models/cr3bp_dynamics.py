"""Circular restricted three-body problem (CR3BP) dynamics.

SYNTHETIC bench: locates the collinear libration point L1 by solving the
equilibrium equation numerically, verifies the Jacobi constant is an
integral of motion along a trajectory, and checks the equilibrium itself
is stationary.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm

_SEED = 20261231 + 923


def accel(state: np.ndarray, mu: float) -> np.ndarray:
    """CR3BP equations in the rotating (synodic) frame, nondimensional.

    state = (x, y, z, vx, vy, vz); primaries at (-mu,0,0) and (1-mu,0,0).
    """
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu) ** 2 + y * y + z * z)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y * y + z * z)
    ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3
    az = -(1 - mu) * z / r1**3 - mu * z / r2**3
    return np.asarray([vx, vy, vz, ax, ay, az], dtype=np.float64)


def jacobi(state: np.ndarray, mu: float) -> float:
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu) ** 2 + y * y + z * z)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y * y + z * z)
    u = 0.5 * (x * x + y * y) + (1 - mu) / r1 + mu / r2
    return float(2 * u - (vx * vx + vy * vy + vz * vz))


def propagate(state: np.ndarray, dt: float, mu: float, nstep: int = 4000) -> np.ndarray:
    s = np.asarray(state, dtype=np.float64).copy()
    h = dt / nstep
    for _ in range(nstep):
        k1 = accel(s, mu)
        k2 = accel(s + h / 2 * k1, mu)
        k3 = accel(s + h / 2 * k2, mu)
        k4 = accel(s + h * k3, mu)
        s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return s


def find_l1(mu: float, tol: float = 1e-12) -> float:
    """L1 x-coordinate via Newton on the planar equilibrium residual."""
    x = 1.0 - mu ** (1 / 3)
    for _ in range(60):
        ax = accel(np.asarray([x, 0.0, 0.0, 0.0, 0.0, 0.0]), mu)[3]
        eps = 1e-7
        dax = (accel(np.asarray([x + eps, 0.0, 0.0, 0.0, 0.0, 0.0]), mu)[3] - ax) / eps
        x -= ax / dax
        if abs(ax) < tol:
            break
    return float(x)


def bench_cr3bp_dynamics(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    mu = 0.01215  # Earth-Moon
    # 1) L1 solve: residual zero
    x1 = find_l1(mu)
    res = float(np.abs(accel(np.asarray([x1, 0.0, 0.0, 0.0, 0.0, 0.0]), mu)[3]))
    score += 1.0 if res < 1e-9 else 0.0
    # 2) L1 between the primaries
    score += 1.0 if -mu < x1 < 1 - mu else 0.0
    # 3) L1 equilibrium is stationary under propagation
    s0 = np.asarray([x1, 0.0, 0.0, 0.0, 0.0, 0.0])
    s1 = propagate(s0, 0.5, mu)
    score += 1.0 if float(norm(s1 - s0)) < 1e-6 else 0.0
    # 4) Jacobi conserved along a non-trivial trajectory
    st = np.asarray([x1 + 0.01, 0.0, 0.0, 0.0, 0.05, 0.0])
    J0 = jacobi(st, mu)
    drift = 0.0
    for _ in range(5):
        st = propagate(st, 0.3, mu)
        drift = max(drift, abs(jacobi(st, mu) - J0))
    score += 1.0 if drift < 1e-6 else 0.0
    rng.random()
    return {"synthetic_cr3bp_dynamics": score / 4.0}
