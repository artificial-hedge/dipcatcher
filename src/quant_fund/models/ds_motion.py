"""Second-order damped dynamical-system motion generator w/ obstacle modulation (SYNTHETIC).

Base field: xdd = k (x* - x) - d xd (critically damped spring-damper, globally
asymptotically stable). Obstacle modulation: velocity is rotated away via a
normal/tangent decomposition inside the influence radius — the SEDS-style
approach — so every start converges to the goal while never entering the
obstacle ball.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 976


def ds_accel(
    x: np.ndarray, xd: np.ndarray, goal: np.ndarray, k: float = 9.0, d: float = 6.0
) -> np.ndarray:
    return np.asarray(k * (goal - x) - d * xd)


def modulate(
    x: np.ndarray, v: np.ndarray, obs: np.ndarray, radius: float, infl: float = 0.45
) -> np.ndarray:
    n = x - obs
    dist = float(np.linalg.norm(n))
    if dist >= radius + infl or dist < 1e-9:
        return np.asarray(v)
    nh = n / dist
    w = (radius + infl - dist) / infl
    vn = float(np.dot(v, nh))
    if vn >= 0:
        return np.asarray(v)
    vt = v - vn * nh
    if np.linalg.norm(vt) < 1e-9:
        vt = np.array([-nh[1], nh[0]])
    vt = vt / np.linalg.norm(vt)
    return np.asarray((1.0 - w) * vn * nh + (np.linalg.norm(v) + w * abs(vn)) * vt)


def simulate(
    x0: np.ndarray, goal: np.ndarray, obs: np.ndarray, radius: float, steps: int = 4000
) -> tuple[np.ndarray, float]:
    x = x0.astype(float).copy()
    v = np.zeros(2)
    dt = 0.01
    min_d = np.inf
    for _ in range(steps):
        v = modulate(x, v + ds_accel(x, v, goal) * dt, obs, radius)
        x = x + v * dt
        min_d = min(min_d, float(np.linalg.norm(x - obs)))
        if np.linalg.norm(x - goal) < 0.02 and np.linalg.norm(v) < 0.05:
            break
    return np.asarray(x), float(min_d)


def bench_ds_motion(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    checks: list[bool] = []
    goal = np.array([1.5, 0.0])
    obs = np.array([0.7, 0.0])
    finals = []
    for ang in rng.uniform(0, 2 * np.pi, 6):
        x0 = 0.4 * np.array([np.cos(ang), np.sin(ang)])
        xf, min_d = simulate(x0, goal, obs, 0.25)
        finals.append(np.linalg.norm(xf - goal))
        checks.append(min_d > 0.24)
    checks.append(max(finals) < 0.05)
    x_far, _ = simulate(np.array([1.5, 1.0]), goal, obs, 0.25)
    checks.append(float(np.linalg.norm(x_far - goal)) < 0.05)
    v_in = modulate(np.array([0.95, 0.0]), np.array([-1.0, 0.0]), obs, 0.25)
    checks.append(float(v_in[0]) >= -0.01)
    v_out = modulate(np.array([1.5, 0.0]), np.array([-1.0, 0.0]), obs, 0.25)
    checks.append(float(v_out[0]) == -1.0)
    score = float(np.mean(checks))
    return {"synthetic_ds_motion": score}
