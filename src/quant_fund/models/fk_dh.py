"""Denavit-Hartenberg forward kinematics vs geometric oracle (wave 283) (SYNTHETIC).

Standard DH transform composition for a 2R planar arm, verified against the
closed-form elbow position formulas.
"""

import numpy as np

_SEED = 20261231 + 782


def _dh(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1],
        ]
    )


def fk2(l1: float, l2: float, th1: float, th2: float) -> np.ndarray:
    t = _dh(l1, 0, 0, th1) @ _dh(l2, 0, 0, th2)
    out: np.ndarray = t[:2, 3]
    return out


def bench_fk_dh(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(10):
        l1, l2 = rng.uniform(0.5, 1.5, 2)
        th1, th2 = rng.uniform(-np.pi, np.pi, 2)
        want = np.array(
            [
                l1 * np.cos(th1) + l2 * np.cos(th1 + th2),
                l1 * np.sin(th1) + l2 * np.sin(th1 + th2),
            ]
        )
        ok += int(np.linalg.norm(fk2(l1, l2, th1, th2) - want) < 1e-9)
    return {"synthetic_fk_dh": float(ok == 10)}
