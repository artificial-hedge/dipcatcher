"""RANSAC plane fit on 3D points vs least-squares-on-outliers oracle (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 701


def ransac_plane(
    pts: np.ndarray, rng: np.random.RandomState, iters: int = 60, tol: float = 0.15
) -> np.ndarray:
    best = np.zeros(4)
    best_n = -1
    n = len(pts)
    for _ in range(iters):
        i = rng.choice(n, 3, replace=False)
        p = pts[i]
        v1, v2 = p[1] - p[0], p[2] - p[0]
        nv = np.cross(v1, v2)
        if np.linalg.norm(nv) < 1e-9:
            continue
        nv = nv / np.linalg.norm(nv)
        d = -nv @ p[0]
        plane = np.append(nv, d)
        inl = np.abs(pts @ nv + d) < tol
        if inl.sum() > best_n:
            best_n = int(inl.sum())
            best = plane
    return best


def bench_ransac_plane(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 30
    for _ in range(trials):
        d = -rng.rand() * 5
        n_in = 60
        inl = np.hstack(
            [rng.rand(n_in, 2) * 10, -d * np.ones((n_in, 1)) + rng.normal(0, 0.02, (n_in, 1))]
        )
        outl = rng.rand(20, 3) * 10
        pts = np.vstack([inl, outl])
        est = ransac_plane(pts, rng)
        if np.linalg.norm(est[:3]) < 1e-9:
            continue
        err = np.abs(pts @ est[:3] + est[3])
        ok += float(err[:n_in].mean() < 0.15)
    return {"synthetic_ransac_inlier_acc": ok / trials}
