"""Stanley cross-track steering controller (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 633


def _stanley_run(path: np.ndarray, k: float, rng: np.random.RandomState) -> float:
    pos = np.array([3.0, -2.0])
    th = 0.3
    err = 0.0
    for _ in range(500):
        d = np.linalg.norm(path - pos, axis=1)
        i = int(np.argmin(d))
        pth = np.arctan2(
            path[min(i + 1, len(path) - 1)][1] - path[i][1],
            path[min(i + 1, len(path) - 1)][0] - path[i][0],
        )
        head_err = (pth - th + np.pi) % (2 * np.pi) - np.pi
        dseg = path[min(i + 1, len(path) - 1)] - path[i]
        cte = float(dseg[0] * (pos - path[i])[1] - dseg[1] * (pos - path[i])[0]) / (
            np.linalg.norm(dseg) + 1e-9
        )
        steer = head_err - np.arctan2(k * cte, 0.4)
        th += 0.25 * steer
        pos += np.array([np.cos(th), np.sin(th)]) * 0.2
        err += abs(d.min())
    return err / 500


def bench_stanley(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    errs = []
    for _ in range(10):
        t = np.linspace(0, 2 * np.pi, 100)
        path = np.stack([5 * np.cos(t), 5 * np.sin(t) + 0.5 * np.sin(3 * t)], axis=1)
        errs.append(_stanley_run(path, 2.5, rng))
    return {"synthetic_stanley_cte": float(np.mean(errs) < 0.5)}
