"""4-point homography via DLT, verified on a synthetic planar pair (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 700


def dlt(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    a = []
    for (x, y), (u, v) in zip(pts1, pts2, strict=True):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.array(a, dtype=float))
    out: np.ndarray = vt[-1].reshape(3, 3)
    return out


def bench_homography_4pt(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 30
    for _ in range(trials):
        h_true = rng.rand(3, 3)
        h_true[2] /= np.linalg.norm(h_true[2]) + 1e-9
        h_true /= h_true[2, 2] + 1e-9
        pts = rng.rand(6, 2) * 10
        ph = h_true @ np.vstack([pts.T, np.ones(6)])
        pts2 = (ph[:2] / ph[2]).T
        h_est = dlt(pts[:4], pts2[:4])
        # evaluate on held-out points
        ph2 = h_est @ np.vstack([pts[4:].T, np.ones(2)])
        reproj = (ph2[:2] / ph2[2]).T
        err = np.linalg.norm(reproj - pts2[4:])
        ok += float(err < 0.5)
    return {"synthetic_homography_reproj": ok / trials}
