"""Luenberger observer on a linear system (wave 279) (SYNTHETIC).

Plant: x' = A x + B u, y = C x. Observer: xh' = A xh + B u + L(y - C xh).
Error dynamics e' = (A - L C) e — placing observer poles left of plant poles
makes the estimate converge, while open-loop dead-reckoning does not.
"""

import numpy as np

_SEED = 20261231 + 758


def _simulate(seed: int, steps: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    a_mat = np.array([[0.0, 1.0], [-2.0, -1.5]])
    b_mat = np.array([[0.0], [1.0]])
    c_vec = np.array([1.0, 0.0])
    # observer gain placing A-LC poles at -3, -4
    l_gain = np.array([[4.5], [5.5]])
    x = rng.normal(0, 2, 2)
    xh = np.zeros(2)
    xs, xhs, drifts = [x.copy()], [xh.copy()], [np.zeros(2)]
    xd = np.zeros(2)
    for t in range(steps):
        u = np.sin(0.05 * t)
        x = x + 0.01 * (a_mat @ x + b_mat[:, 0] * u)
        y = float(c_vec @ x)
        xh = xh + 0.01 * (a_mat @ xh + b_mat[:, 0] * u + l_gain[:, 0] * (y - float(c_vec @ xh)))
        xd = xd + 0.01 * (a_mat @ xd + b_mat[:, 0] * u)  # dead reckoning
        xs.append(x.copy())
        xhs.append(xh.copy())
        drifts.append(xd.copy())
    return np.array(xs), np.array(xhs), np.array(drifts)


def bench_luen_obsv(seed: int = _SEED) -> dict[str, float]:
    xs, xhs, drifts = _simulate(seed)
    err_o = float(np.abs(xs[-50:] - xhs[-50:]).mean())
    err_d = float(np.abs(xs[-50:] - drifts[-50:]).mean())
    return {
        "synthetic_luen_converge": float(err_o < 0.05),
        "synthetic_luen_better": float(err_o < 0.2 * err_d),
    }
