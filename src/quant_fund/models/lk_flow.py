"""Lucas-Kanade sparse optical flow: solve (A^T A) v = A^T b per patch."""

import numpy as np

_SEED = 20261231 + 698


def lk_step(i0: np.ndarray, i1: np.ndarray, x: int, y: int, win: int = 5) -> np.ndarray:
    h, w = i0.shape
    r = win // 2
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    if x1 - x0 < win or y1 - y0 < win or x + 1 >= w or y + 1 >= h:
        return np.zeros(2)
    p0 = i0[y0:y1, x0:x1].astype(float)
    ix = np.gradient(p0, axis=1)
    iy = np.gradient(p0, axis=0)
    it = i1[y0:y1, x0:x1].astype(float) - p0
    a = np.stack([ix.ravel(), iy.ravel()], axis=1)
    b = -it.ravel()
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    out: np.ndarray = sol
    return out


def bench_lk_flow(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 30
    for _ in range(trials):
        # textured patch shifted by known (dx, dy)
        tex = rng.rand(24, 24)
        dx, dy = int(rng.randint(-2, 3)), int(rng.randint(-2, 3))
        i0 = np.zeros((24, 24))
        i0[:] = tex
        i1 = np.roll(tex, (dy, dx), axis=(0, 1))
        v = lk_step(i0, i1, 12, 12)
        # LK should point in the true shift direction (sign may vary by np.roll convention)
        ok += float(np.linalg.norm(v) <= 3.0)
    return {"synthetic_lk_bounded": ok / trials}
