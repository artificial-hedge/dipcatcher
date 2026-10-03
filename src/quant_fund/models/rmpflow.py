"""RMPflow-lite: task-space Riemannian motion policies pulled into joint space.

Leaf RMPs (goal attractor + distance-capped obstacle repulsor) are fused by
metric-weighted average in task space and pulled back to configuration space
through the task Jacobian: qdd = J^+ (f_task - Jdot qd). A 2-link planar arm
must reach a goal while the repulsive leaf dominates near the obstacle.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 975


def planar_arm_fk(q: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    a1, a2 = lengths
    return np.array(
        [
            a1 * np.cos(q[0]) + a2 * np.cos(q[0] + q[1]),
            a1 * np.sin(q[0]) + a2 * np.sin(q[0] + q[1]),
        ]
    )


def planar_arm_jac(q: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    a1, a2 = lengths
    s = q[0] + q[1]
    return np.array(
        [
            [-a1 * np.sin(q[0]) - a2 * np.sin(s), -a2 * np.sin(s)],
            [a1 * np.cos(q[0]) + a2 * np.cos(s), a2 * np.cos(s)],
        ]
    )


def goal_rmp(x: np.ndarray, xd: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eta, zeta, eps = 8.0, 4.0, 1e-3
    d = np.linalg.norm(goal - x) + eps
    w = 1.0 / (1.0 + 4.0 * d)
    f = eta * (goal - x) - zeta * xd
    return np.asarray(f), np.asarray(w * np.eye(2))


def obstacle_rmp(
    x: np.ndarray, xd: np.ndarray, obs: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    d = np.linalg.norm(x - obs) - radius
    if d > 0.25:
        return np.zeros(2), np.zeros((2, 2))
    w = np.exp(-8.0 * d)
    f = 25.0 * w * (x - obs) / (np.linalg.norm(x - obs) + 1e-9) - 2.0 * xd
    return np.asarray(f), np.asarray(8.0 * w * np.eye(2))


def rmp_step(
    q: np.ndarray, qd: np.ndarray, goal: np.ndarray, obs: np.ndarray, radius: float, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.array([1.0, 1.0])
    x = planar_arm_fk(q, lengths)
    j = planar_arm_jac(q, lengths)
    f_g, m_g = goal_rmp(x, j @ qd, goal)
    f_o, m_o = obstacle_rmp(x, j @ qd, obs, radius)
    num = m_g @ f_g + m_o @ f_o
    den = m_g + m_o + 1e-9 * np.eye(2)
    f_task = np.linalg.solve(den, num)
    qdd = np.linalg.pinv(j) @ f_task - 1.5 * qd
    qd = np.asarray(qd + dt * qdd)
    q = np.asarray(q + dt * qd)
    return q, qd, x


def bench_rmpflow(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    checks: list[bool] = []
    goal = np.array([0.5, 0.3])
    obs = np.array([1.0, 0.6])
    q = np.array([0.2, 0.6])
    qd = np.zeros(2)
    min_d = np.inf
    for _ in range(6000):
        q, qd, x = rmp_step(q, qd, goal, obs, 0.15, 0.005)
        min_d = min(min_d, float(np.linalg.norm(x - obs)))
    checks.append(float(np.linalg.norm(x - goal)) < 0.05)
    checks.append(min_d > 0.15)
    f_o, m_o = obstacle_rmp(np.array([1.0, 0.8]), np.zeros(2), obs, 0.15)
    checks.append(float(m_o[0, 0]) > 3.0 and float(f_o[1]) > 0.0)
    f_c = obstacle_rmp(np.array([1.0, 0.6]), np.zeros(2), obs, 0.15)
    checks.append(bool(np.isfinite(f_c[0]).all()))
    f_o_far, m_o_far = obstacle_rmp(np.array([1.8, 1.2]), np.zeros(2), obs, 0.15)
    checks.append(float(m_o_far[0, 0]) == 0.0)
    _ = rng
    score = float(np.mean(checks))
    return {"synthetic_rmpflow": score}
