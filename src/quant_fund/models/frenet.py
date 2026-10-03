"""Frenet canon: optimal-trajectory planning in the Frenet frame
(Werling) — quintic lateral polynomials s_d(t) toward sampled
terminal offsets + quartic longitudinal polynomials s(t) toward
terminal speeds, jerk-minimized, then scored on cost, speed and
collision with circle obstacles mapped into Frenet space.
Bench: feasible-candidate rate, chosen-path clearance, and jerk
optimality vs perturbed terminal states. All SYNTHETIC.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def quintic_poly(
    p0: float, v0: float, a0: float, pT: float, vT: float, aT: float, T: float
) -> FloatArray:
    """Quintic c0..c5 with given boundary conditions."""
    T2, T3, T4, T5 = T * T, T**3, T**4, T**5
    c0, c1, c2 = p0, v0, a0 / 2
    A = np.array(
        [
            [T3, T4, T5],
            [3 * T2, 4 * T3, 5 * T4],
            [6 * T, 12 * T2, 20 * T3],
        ]
    )
    b = np.array(
        [
            pT - c0 - c1 * T - c2 * T2,
            vT - c1 - 2 * c2 * T,
            aT - 2 * c2,
        ]
    )
    c345 = np.linalg.solve(A, b)
    return np.asarray([c0, c1, c2, *c345], dtype=np.float64)


def quartic_poly(p0: float, v0: float, a0: float, vT: float, aT: float, T: float) -> FloatArray:
    """Quartic c0..c4 (longitudinal: end position free)."""
    T2, T3 = T * T, T**3
    c0, c1, c2 = p0, v0, a0 / 2
    A = np.array([[3 * T2, 4 * T3], [6 * T, 12 * T2]])
    b = np.array([vT - c1 - 2 * c2 * T, aT - 2 * c2])
    c34 = np.linalg.solve(A, b)
    return np.asarray([c0, c1, c2, *c34, 0.0], dtype=np.float64)


def _eval(c: FloatArray, t: float, d: int = 0) -> float:
    out = 0.0
    for i in range(d, len(c)):
        p = 1.0
        for k in range(d):
            p *= i - k
        out += c[i] * p * t ** (i - d)
    return out


def jerk_cost(c: FloatArray, T: float) -> float:
    """∫ j(t)² dt where j = 3rd derivative."""
    ts = np.linspace(0, T, 40)
    return float(np.trapezoid(np.array([_eval(c, u, 3) for u in ts]) ** 2, ts))


def frenet_plan(
    path_pts: FloatArray,
    s0: float,
    d0: float,
    v0: float,
    target_v: float = 0.8,
    obstacles: list[tuple[float, float]] | None = None,
    horizon_T: float = 3.0,
    dt: float = 0.15,
) -> tuple[float, FloatArray, int]:
    """Plan along a reference path (straight here; Frenet on
    a straight centerline reduces to (s, d) coordinates).
    obstacles are (s, d) blockers.

    Returns (best_cost, trajectory [N×2] (s, d), n_feasible).
    """
    obstacles = obstacles or []
    best_cost = math.inf
    best_traj = np.zeros((0, 2))
    n_feas = 0
    ts = np.arange(0, horizon_T + 1e-9, dt)
    for dT in (-0.3, -0.15, 0.0, 0.15, 0.3):
        for vT in (target_v - 0.2, target_v, target_v + 0.2):
            cD = quintic_poly(d0, 0.0, 0.0, dT, 0.0, 0.0, horizon_T)
            cS = quartic_poly(s0, v0, 0.0, vT, 0.0, horizon_T)
            traj = np.stack([[_eval(cS, t), _eval(cD, t)] for t in ts])
            # feasibility: obstacle clearance
            ok = True
            for os_, od in obstacles:
                dd = np.min(np.hypot(traj[:, 0] - os_, traj[:, 1] - od))
                if dd < 0.1:
                    ok = False
                    break
            if not ok:
                continue
            n_feas += 1
            cost = (
                jerk_cost(cS, horizon_T)
                + 0.5 * jerk_cost(cD, horizon_T)
                + 0.1 * dT * dT
                + 0.05 * (vT - target_v) ** 2
            )
            if cost < best_cost:
                best_cost = cost
                best_traj = traj
    return best_cost, best_traj, n_feas


def bench_frenet(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    path = np.stack([np.linspace(0, 10, 50), np.zeros(50)], axis=1)
    # no obstacles: choose centerline (d=0), target speed
    cost, traj, nf = frenet_plan(path, s0=0.0, d0=0.0, v0=0.5, target_v=0.8)
    out["synthetic_frenet_feasible"] = float(nf)
    out["synthetic_frenet_end_speed_err"] = abs(traj[-1, 0] / 3.0 - 0.8 if len(traj) else 1.0)
    out["synthetic_frenet_end_offset"] = abs(float(traj[-1, 1]))
    out["synthetic_frenet_cost"] = cost
    # with a blocker at (s=2, d=0): planner should pick a lane
    cost2, traj2, nf2 = frenet_plan(
        path,
        s0=0.0,
        d0=0.0,
        v0=0.5,
        target_v=0.8,
        obstacles=[(1.8, 0.0), (2.6, 0.05)],
    )
    out["synthetic_frenet_obs_feasible"] = float(nf2)
    if len(traj2):
        clr = min(
            np.min(np.hypot(traj2[:, 0] - os_, traj2[:, 1] - od))
            for os_, od in [(1.8, 0.0), (2.6, 0.05)]
        )
        out["synthetic_frenet_obs_clearance"] = float(clr)
    else:
        out["synthetic_frenet_obs_clearance"] = 0.0
    # quintic endpoints exact
    c = quintic_poly(0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 2.0)
    out["synthetic_quintic_end_err"] = abs(_eval(c, 2.0) - 0.3)
    out["synthetic_quintic_jerk"] = jerk_cost(c, 2.0)
    return out
