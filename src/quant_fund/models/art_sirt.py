"""ART / SIRT iterative tomographic reconstruction.

The forward projector is a sparse ray-sum matrix A (each row sums the
pixels along one line through the image). ART (Kaczmarz) updates one row
at a time; SIRT does simultaneous Landweber updates weighted by row/column
norms. Bench verifies the data residual and image error both decrease.
"""

import numpy as np

_SEED = 20261231 + 873


def _line_pixels(n: int, x0: float, y0: float, x1: float, y1: float) -> list[int]:
    pts = []
    steps = int(2 * n)
    for t in np.linspace(0, 1, steps):
        x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < n and 0 <= iy < n:
            pts.append(iy * n + ix)
    return pts


def _projector(n: int, n_rays: int, n_angles: int, rng: np.random.Generator) -> np.ndarray:
    rows = []
    for _ in range(n_angles):
        th = rng.uniform(0, np.pi)
        for _ in range(n_rays):
            c = (n - 1) / 2.0
            off = rng.uniform(-c, c)
            dx, dy = np.cos(th), np.sin(th)
            px, py = -dy, dx
            row = np.zeros(n * n)
            row[
                _line_pixels(
                    n,
                    c + off * px - c * dx,
                    c + off * py - c * dy,
                    c + off * px + c * dx,
                    c + off * py + c * dy,
                )
            ] = 1.0
            rows.append(row)
    return np.array(rows)


def art(A: np.ndarray, b: np.ndarray, iters: int = 20, lam: float = 0.3) -> np.ndarray:
    x = np.zeros(A.shape[1])
    norms = (A**2).sum(axis=1) + 1e-9
    for _ in range(iters):
        for i in range(A.shape[0]):
            x += lam * (b[i] - A[i] @ x) / norms[i] * A[i]
    return np.maximum(x, 0.0)


def sirt(A: np.ndarray, b: np.ndarray, iters: int = 40) -> np.ndarray:
    x = np.zeros(A.shape[1])
    row_w = 1.0 / (A.sum(axis=1) + 1e-9)
    col_w = 1.0 / (A.sum(axis=0) + 1e-9)
    for _ in range(iters):
        x += col_w * (A.T @ (row_w * (b - A @ x)))
    return np.maximum(x, 0.0)


def bench_art_sirt(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: residual + image error fall with iterations; SIRT beats 1-step."""
    rng = np.random.default_rng(seed)
    n = 16
    yy, xx = np.mgrid[0:n, 0:n] / n - 0.5
    img = ((xx / 0.3) ** 2 + (yy / 0.3) ** 2 <= 1).astype(float).ravel()
    A = _projector(n, 10, 10, rng)
    b = A @ img
    x1 = sirt(A, b, iters=1)
    x40 = sirt(A, b, iters=40)
    res1 = np.linalg.norm(A @ x1 - b)
    res40 = np.linalg.norm(A @ x40 - b)
    err1 = np.linalg.norm(x1 - img)
    err40 = np.linalg.norm(x40 - img)
    xa = art(A, b, iters=8)
    err_art = np.linalg.norm(xa - img)
    ok = res40 < res1 and err40 < err1 and err_art < err1
    return {"synthetic_art_sirt": 1.0 if ok else 0.0}
