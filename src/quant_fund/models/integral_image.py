"""Integral image (summed-area table) + box filters and Haar-like features.

SYNTHETIC bench only.
"""

import numpy as np

_SEED = 20261231 + 910


def integral(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float64)
    return np.asarray(np.cumsum(np.cumsum(img, axis=0), axis=1), dtype=np.float64)


def box_sum(ii: np.ndarray, y0: int, x0: int, h: int, w: int) -> float:
    """Sum over rows [y0, y0+h), cols [x0, x0+w) via 4 lookups."""
    y1, x1 = y0 + h - 1, x0 + w - 1
    a = ii[y1, x1]
    b = ii[y0 - 1, x1] if y0 > 0 else 0.0
    c = ii[y1, x0 - 1] if x0 > 0 else 0.0
    d = ii[y0 - 1, x0 - 1] if y0 > 0 and x0 > 0 else 0.0
    return float(a - b - c + d)


def box_filter(img: np.ndarray, k: int) -> np.ndarray:
    ii = integral(np.pad(img, ((k // 2 + 1, k // 2), (k // 2 + 1, k // 2))))
    n, m = img.shape
    out = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            out[i, j] = box_sum(ii, i + 1, j + 1, k, k) / (k * k)
    return np.asarray(out, dtype=np.float64)


def haar_two_rect(ii: np.ndarray, y0: int, x0: int, h: int, w: int) -> float:
    """Vertical two-rectangle Haar feature: left minus right."""
    return box_sum(ii, y0, x0, h, w // 2) - box_sum(ii, y0, x0 + w // 2, h, w // 2)


def bench_integral_image(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    img = rng.uniform(0, 1, (40, 50))
    ii = integral(img)
    # random windows vs brute force
    ok = True
    for _ in range(200):
        y0 = int(rng.integers(0, 30))
        x0 = int(rng.integers(0, 40))
        h = int(rng.integers(1, 10))
        w = int(rng.integers(1, 10))
        truth = float(img[y0 : y0 + h, x0 : x0 + w].sum())
        ok = ok and abs(box_sum(ii, y0, x0, h, w) - truth) < 1e-9
    score += 1.0 if ok else 0.0
    # box filter matches convolution on interior
    bf = box_filter(img, 5)
    truth5 = np.array(
        [[img[i - 2 : i + 3, j - 2 : j + 3].mean() for j in range(2, 48)] for i in range(2, 38)]
    )
    score += 1.0 if np.max(np.abs(bf[2:38, 2:48] - truth5)) < 1e-9 else 0.0
    # Haar feature fires on an edge: bright left, dark right
    img2 = np.zeros((20, 20))
    img2[:, :10] = 1.0
    ii2 = integral(img2)
    f_edge = haar_two_rect(ii2, 4, 2, 12, 16)
    img3 = np.zeros((20, 20))
    ii3 = integral(img3)
    f_flat = haar_two_rect(ii3, 4, 2, 12, 16)
    score += 1.0 if f_edge > 80 and abs(f_flat) < 1e-9 else 0.0
    # full-image sum = last cell
    score += 1.0 if abs(ii[-1, -1] - img.sum()) < 1e-9 else 0.0
    return {"synthetic_integral_image": score / 4.0}
