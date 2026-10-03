"""DWA canon: dynamic-window approach for a unicycle robot —
sample (v, ω) inside the reachable velocity window, roll out a
short arc, score by heading-to-goal + clearance + velocity, pick
the argmax each cycle. Bench: goal reaching, minimum clearance,
and time-to-goal against a bug-style naive steering baseline.
All SYNTHETIC.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _min_obstacle_dist(x: float, y: float, obstacles: list[tuple[float, float, float]]) -> float:
    if not obstacles:
        return math.inf
    return min(math.hypot(x - ox, y - oy) - r for ox, oy, r in obstacles)


def _rollout(
    pose: tuple[float, float, float],
    v: float,
    w: float,
    dt: float,
    horizon: float,
) -> FloatArray:
    x, y, th = pose
    pts = []
    t = 0.0
    while t < horizon:
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        th += w * dt
        pts.append((x, y))
        t += dt
    return np.asarray(pts, dtype=np.float64)


def dwa_drive(
    start: tuple[float, float, float],
    goal: tuple[float, float],
    obstacles: list[tuple[float, float, float]],
    v_max: float = 0.5,
    w_max: float = 1.5,
    acc_v: float = 0.4,
    acc_w: float = 2.0,
    dt: float = 0.1,
    horizon: float = 1.2,
    goal_tol: float = 0.08,
    max_steps: int = 600,
) -> tuple[bool, int, float]:
    """DWA local planner loop.

    Returns (reached, steps, min_clearance_over_run).
    """
    pose = start
    v, w = 0.0, 0.0
    min_clear = math.inf
    for step in range(max_steps):
        x, y, th = pose
        if math.hypot(goal[0] - x, goal[1] - y) < goal_tol:
            return True, step, min_clear
        # dynamic window
        v_lo = max(0.0, v - acc_v * dt)
        v_hi = min(v_max, v + acc_v * dt)
        w_lo = max(-w_max, w - acc_w * dt)
        w_hi = min(w_max, w + acc_w * dt)
        d0 = math.hypot(goal[0] - x, goal[1] - y)
        best = None
        best_score = -math.inf
        for vv in np.linspace(v_lo, v_hi, 7):
            for ww in np.linspace(w_lo, w_hi, 13):
                pts = _rollout(pose, vv, ww, dt, horizon)
                # collision check
                clear = min(_min_obstacle_dist(px, py, obstacles) for px, py in pts)
                if clear < 0:
                    continue
                # progress toward the goal at the arc end
                hx, hy = pts[-1]
                d_end = math.hypot(goal[0] - hx, goal[1] - hy)
                progress = (d0 - d_end) / (v_max * horizon)
                score = 0.55 * progress + 0.25 * min(clear, 0.5) / 0.5 + 0.2 * vv / v_max
                if score > best_score:
                    best_score = score
                    best = (vv, ww)
        if best is None:
            # trapped — stop
            return False, step, min_clear
        v, w = best
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        th += w * dt
        pose = (x, y, th)
        clear_now = _min_obstacle_dist(x, y, obstacles)
        min_clear = min(min_clear, clear_now)
        if clear_now < 0:
            return False, step, min_clear
    return False, max_steps, min_clear


def naive_drive(
    start: tuple[float, float, float],
    goal: tuple[float, float],
    obstacles: list[tuple[float, float, float]],
    v: float = 0.4,
    dt: float = 0.1,
    max_steps: int = 600,
) -> tuple[bool, int, float]:
    """Naive steer-to-goal (no obstacle handling) — baseline."""
    x, y, th = start
    min_clear = math.inf
    for step in range(max_steps):
        if math.hypot(goal[0] - x, goal[1] - y) < 0.08:
            return True, step, min_clear
        th = math.atan2(goal[1] - y, goal[0] - x)
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        c = _min_obstacle_dist(x, y, obstacles)
        min_clear = min(min_clear, c)
        if c < 0:
            return False, step, min_clear
    return False, max_steps, min_clear


def bench_dwa(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    obstacles = [(0.5, 0.45, 0.1), (0.55, 0.65, 0.08)]
    reached, steps, clear = dwa_drive((0.1, 0.1, 0.0), (0.9, 0.9), obstacles)
    out["synthetic_dwa_reached"] = float(reached)
    out["synthetic_dwa_steps"] = float(steps)
    out["synthetic_dwa_min_clearance"] = clear
    # naive baseline hits the obstacle
    nreached, nsteps, nclear = naive_drive((0.1, 0.1, 0.0), (0.9, 0.9), obstacles)
    out["synthetic_naive_reached"] = float(nreached)
    out["synthetic_naive_min_clearance"] = nclear
    out["synthetic_dwa_vs_naive"] = out["synthetic_dwa_reached"] - out["synthetic_naive_reached"]
    # corridor: reachable but narrow
    obs2 = [(0.4, 0.42, 0.08), (0.4, 0.58, 0.08), (0.6, 0.35, 0.07)]
    r2, s2, c2 = dwa_drive((0.1, 0.1, 0.0), (0.9, 0.6), obs2)
    out["synthetic_dwa_corridor_reached"] = float(r2)
    out["synthetic_dwa_corridor_clearance"] = c2
    return out
