"""Prioritized task-stack whole-body control via null-space projection (SYNTHETIC).

Task j supplies (J_j, target_j). Descending priority: qdd solves each task
exactly in the orthogonal complement of all higher-priority task Jacobians —
the classic operational-space priority scheme (Sentis & Khatib). Bench: a
3-DoF planar arm must place the end effector (priority 1) exactly while the
posture task (priority 2) improves vs an unprioritized damped least-squares
blend.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 977


def solve_prioritized(tasks: list[tuple[np.ndarray, np.ndarray]], damp: float = 1e-6) -> np.ndarray:
    n = tasks[0][0].shape[1]
    qdd = np.zeros(n)
    n_acc = np.eye(n)
    for j, t in tasks:
        jp = j @ n_acc
        aug = jp @ jp.T + damp * np.eye(jp.shape[0])
        step = jp.T @ np.linalg.solve(aug, t - j @ qdd)
        qdd = qdd + n_acc @ step
        n_acc = n_acc @ (np.eye(n) - np.linalg.pinv(jp) @ jp)
    return np.asarray(qdd)


def solve_blend(tasks: list[tuple[np.ndarray, np.ndarray]], w: float = 0.5) -> np.ndarray:
    j1, t1 = tasks[0]
    j2, t2 = tasks[1]
    n = j1.shape[1]
    g = np.zeros(n)
    g += np.linalg.pinv(j1.T @ j1 + 1e-4 * np.eye(n)) @ j1.T @ t1 * (1 - w)
    g += np.linalg.pinv(j2.T @ j2 + 1e-4 * np.eye(n)) @ j2.T @ t2 * w
    return np.asarray(g)


def bench_wbc_qp(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    checks: list[bool] = []
    j1 = rng.normal(size=(2, 3))
    j2 = np.eye(3)
    t1 = rng.normal(size=2)
    t2 = rng.normal(size=3)
    q = solve_prioritized([(j1, t1), (j2, t2)])
    checks.append(float(np.linalg.norm(j1 @ q - t1)) < 1e-3)
    kkt = np.block([[j2.T @ j2, j1.T], [j1, np.zeros((2, 2))]])
    rhs = np.concatenate([j2.T @ t2, t1])
    q_star = np.linalg.solve(kkt, rhs)[:3]
    checks.append(float(np.linalg.norm(q - q_star)) < 1e-3)
    q3 = solve_prioritized([(j1, t1)])
    checks.append(float(np.linalg.norm(j1 @ q3 - t1)) < 1e-3)
    j_sing = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    qs = solve_prioritized([(j_sing, np.array([1.0, 2.0])), (j2, t2)])
    checks.append(bool(np.isfinite(qs).all()) and abs(qs[0] - 1.0) < 1e-2)
    score = float(np.mean(checks))
    return {"synthetic_wbc_qp": score}
