"""8-point essential/fundamental matrix; epipolar-constraint residual."""

import numpy as np

_SEED = 20261231 + 702


def fundamental(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    a = []
    for (x, y), (u, v) in zip(pts1, pts2, strict=True):
        a.append([u * x, u * y, u, v * x, v * y, v, x, y, 1.0])
    _, _, vt = np.linalg.svd(np.array(a, dtype=float))
    f = vt[-1].reshape(3, 3)
    # enforce rank-2
    u, s, vt2 = np.linalg.svd(f)
    s[2] = 0
    out: np.ndarray = u @ np.diag(s) @ vt2
    return out


def bench_epipolar_8pt(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 30
    for _ in range(trials):
        # project synthetic 3D points through two pinhole cameras
        pts3 = rng.rand(12, 3) * 10
        pts3[:, 2] += 5
        k = np.array([[100.0, 0, 50], [0, 100, 50], [0, 0, 1]])
        r2 = np.eye(3)
        t2 = np.array([0.5, 0.0, 0.0])
        p1 = k @ pts3.T
        p2 = k @ (r2 @ pts3.T + t2[:, None])
        uv1 = (p1[:2] / p1[2]).T
        uv2 = (p2[:2] / p2[2]).T
        f = fundamental(uv1[:8], uv2[:8])
        # check epipolar residuals on held-out
        resid = np.abs(
            np.sum(
                np.vstack([uv2[8:].T, np.ones(4)]) * (f @ np.vstack([uv1[8:].T, np.ones(4)])),
                axis=0,
            )
        )
        ok += float(resid.max() < 0.5)
    return {"synthetic_epipolar_residual": ok / trials}
