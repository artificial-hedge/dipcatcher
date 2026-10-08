"""Jacobian damped-least-squares inverse kinematics (wave 283) (SYNTHETIC).

Iterate theta += J^T (J J^T + lam^2 I)^{-1} * err for the planar 2R arm;
converges to reachable targets and reports failure on unreachable ones.
"""

import numpy as np

from quant_fund.models.fk_dh import fk2

_SEED = 20261231 + 783


def ik2(l1: float, l2: float, target: np.ndarray, iters: int = 500) -> tuple[np.ndarray, bool]:
    th = np.array([0.1, 0.1])
    lam = 0.1
    for _ in range(iters):
        pos = fk2(l1, l2, th[0], th[1])
        err = target - pos
        if np.linalg.norm(err) < 1e-6:
            return th, True
        j1 = np.array(
            [
                -l1 * np.sin(th[0]) - l2 * np.sin(th[0] + th[1]),
                l1 * np.cos(th[0]) + l2 * np.cos(th[0] + th[1]),
            ]
        )
        j2 = np.array([-l2 * np.sin(th[0] + th[1]), l2 * np.cos(th[0] + th[1])])
        jmat = np.stack([j1, j2], 1)
        d_th = jmat.T @ np.linalg.solve(jmat @ jmat.T + lam**2 * np.eye(2), err)
        th = th + d_th
    return th, False


def bench_ik_jac(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    l1 = l2 = 1.0
    for _ in range(8):
        # sample reachable targets via FK of random angles
        th = rng.uniform(-1.2, 1.2, 2)
        tgt = fk2(l1, l2, th[0], th[1])
        th_sol, conv = ik2(l1, l2, tgt)
        got = fk2(l1, l2, th_sol[0], th_sol[1])
        ok += int(conv and np.linalg.norm(got - tgt) < 1e-4)
    unreachable = np.array([3.5, 0.0])
    _, conv2 = ik2(l1, l2, unreachable, iters=200)
    ok += int(
        not conv2
        or np.linalg.norm(fk2(l1, l2, *ik2(l1, l2, unreachable, iters=200)[0]) - unreachable) > 0.1
    )
    return {"synthetic_ik_jac": float(ok >= 8)}
