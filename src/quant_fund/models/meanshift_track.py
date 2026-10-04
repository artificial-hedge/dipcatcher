"""Mean-shift mode seeking + simple object tracking by histogram back-projection.

SYNTHETIC bench only.
"""

import numpy as np

_SEED = 20261231 + 913


def mean_shift(pts: np.ndarray, x0: np.ndarray, h: float, iters: int = 50) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    x = np.asarray(x0, dtype=np.float64).copy()
    for _ in range(iters):
        d2 = np.sum((pts - x) ** 2, axis=1)
        w = np.exp(-d2 / (2 * h * h))
        x_new = (w[:, None] * pts).sum(axis=0) / max(w.sum(), 1e-12)
        if np.linalg.norm(x_new - x) < 1e-6:
            x = x_new
            break
        x = x_new
    return np.asarray(x, dtype=np.float64)


def back_project(
    img: np.ndarray,
    model_hist: np.ndarray,
    lo: float,
    hi: float,
    nbins: int = 16,
) -> np.ndarray:
    """Back-project model histogram onto img; lo/hi are the model's binning range."""
    img = np.asarray(img, dtype=np.float64)
    idx = np.clip(((img - lo) / (hi - lo) * nbins).astype(int), 0, nbins - 1)
    prob = model_hist[idx] / (model_hist.max() + 1e-9)
    return np.asarray(prob, dtype=np.float64)


def model_hist(patch: np.ndarray, lo: float, hi: float, nbins: int = 16) -> np.ndarray:
    patch = np.asarray(patch, dtype=np.float64)
    idx = np.clip(((patch - lo) / (hi - lo) * nbins).astype(int), 0, nbins - 1)
    h = np.bincount(idx.ravel(), minlength=nbins).astype(np.float64)
    return np.asarray(h / max(h.sum(), 1e-9), dtype=np.float64)


def track_step(prob: np.ndarray, center: tuple[float, float], half: int) -> tuple[float, float]:
    """Mean-shift the window center on the back-projection image."""
    ys, xs = np.mgrid[0 : prob.shape[0], 0 : prob.shape[1]]
    cy, cx = center
    for _ in range(20):
        sel = (np.abs(ys - cy) <= half) & (np.abs(xs - cx) <= half)
        w = prob * sel
        s = w.sum()
        if s <= 1e-9:
            break
        ny = float((w * ys).sum() / s)
        nx = float((w * xs).sum() / s)
        if abs(ny - cy) < 1e-3 and abs(nx - cx) < 1e-3:
            cy, cx = ny, nx
            break
        cy, cx = ny, nx
    return (cy, cx)


def bench_meanshift_track(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    # 2-mode cluster: start near each → converge to its mode
    pts = np.vstack([rng.normal([0.0, 0.0], 0.5, (200, 2)), rng.normal([8.0, 8.0], 0.5, (200, 2))])
    m1 = mean_shift(pts, np.array([1.0, 0.5]), 1.0)
    m2 = mean_shift(pts, np.array([7.0, 8.5]), 1.0)
    score += 1.0 if np.linalg.norm(m1 - [0, 0]) < 0.4 else 0.0
    score += 1.0 if np.linalg.norm(m2 - [8, 8]) < 0.4 else 0.0
    # tracking: bright blob moves; tracker follows via back-projection
    n = 60
    img = rng.normal(0.2, 0.05, (n, n))
    img[10:18, 10:18] = 1.0
    lo, hi = 0.0, 1.0
    hist = model_hist(img[10:18, 10:18], lo, hi)
    prob = back_project(img, hist, lo, hi)
    c = track_step(prob, (14.0, 14.0), 6)
    score += 1.0 if abs(c[0] - 13.5) < 3 and abs(c[1] - 13.5) < 3 else 0.0
    # blob moves 15px → same model finds it
    img2 = rng.normal(0.2, 0.05, (n, n))
    img2[25:33, 25:33] = 1.0
    prob2 = back_project(img2, hist, lo, hi)
    c2 = track_step(prob2, (14.0, 14.0), 16)
    score += 1.0 if abs(c2[0] - 28.5) < 4 and abs(c2[1] - 28.5) < 4 else 0.0
    return {"synthetic_meanshift_track": score / 4.0}
