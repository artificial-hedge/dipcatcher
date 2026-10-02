"""Occupancy-grid mapping via log-odds inverse sensor model."""

import numpy as np

_SEED = 20261231 + 631


def _map_sensor(
    truth: np.ndarray, poses: np.ndarray, rng: np.random.RandomState, fov: float = 6.0
) -> np.ndarray:
    n = truth.shape[0]
    logodds = np.zeros((n, n))
    l_occ, l_free = 0.85, -0.4
    for px, py in poses.astype(int):
        for i in range(max(0, px - 6), min(n, px + 7)):
            for j in range(max(0, py - 6), min(n, py + 7)):
                d = np.hypot(i - px, j - py)
                if d > fov:
                    continue
                obs = truth[i, j] + rng.normal(0, 0.1)
                logodds[i, j] += l_occ if obs > 0.5 else l_free
    return 1.0 / (1.0 + np.exp(-logodds))


def bench_occupancy_grid(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    accs = []
    for _ in range(8):
        n = 40
        truth = (rng.rand(n, n) < 0.3).astype(float)
        truth[0, :] = truth[-1, :] = truth[:, 0] = truth[:, -1] = 1.0
        poses = rng.randint(4, n - 4, (30, 2))
        prob = _map_sensor(truth, poses, rng)
        est = prob > 0.5
        # only cells visited count
        seen = np.zeros((n, n), bool)
        for px, py in poses:
            seen[max(0, px - 6) : px + 7, max(0, py - 6) : py + 7] = True
        accs.append(float(np.mean(est[seen] == truth[seen])))
    return {"synthetic_map_accuracy": float(np.mean(accs))}
