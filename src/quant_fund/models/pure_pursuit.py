"""Pure-pursuit path tracking vs naive heading controller (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 632


def _track(path: np.ndarray, lookahead: float, rng: np.random.RandomState, mode: str) -> float:
    pos = np.array([0.0, -1.0])
    th = np.pi / 4
    err = 0.0
    steps = 400
    for _ in range(steps):
        d = np.linalg.norm(path - pos, axis=1)
        if mode == "pure":
            i = int(np.argmax(d > lookahead)) if (d > lookahead).any() else len(path) - 1
            tgt = path[i]
        else:
            tgt = path[int(np.argmin(d))]
        ang = np.arctan2(tgt[1] - pos[1], tgt[0] - pos[0])
        steer = (ang - th + np.pi) % (2 * np.pi) - np.pi
        th += 0.3 * steer + rng.normal(0, 0.005)
        pos += np.array([np.cos(th), np.sin(th)]) * 0.2
        err += d.min()
    return err / steps


def bench_pure_pursuit(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    wins = 0
    for _ in range(12):
        t = np.linspace(0, 2 * np.pi, 60)
        path = np.stack([3 * np.cos(t), 2 * np.sin(t) + 0.3 * np.sin(3 * t)], axis=1)
        e_pp = _track(path, 1.0, rng, "pure")
        e_nv = _track(path, 1.0, rng, "naive")
        wins += e_pp <= e_nv + 0.5
    return {"synthetic_pp_win": wins / 12}
