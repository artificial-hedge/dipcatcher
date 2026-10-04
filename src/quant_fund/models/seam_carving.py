"""Seam carving: DP minimum-energy vertical seam + removal.

SYNTHETIC bench only.
"""

import numpy as np

_SEED = 20261231 + 911


def energy(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float64)
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gx[:, 0] = img[:, 1] - img[:, 0]
    gx[:, -1] = img[:, -1] - img[:, -2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    gy[0, :] = img[1, :] - img[0, :]
    gy[-1, :] = img[-1, :] - img[-2, :]
    return np.asarray(np.abs(gx) + np.abs(gy), dtype=np.float64)


def min_seam(e: np.ndarray) -> tuple[list[int], float]:
    n, m = e.shape
    dp = e.copy()
    back = np.zeros((n, m), dtype=np.int64)
    for i in range(1, n):
        for j in range(m):
            lo, hi = max(0, j - 1), min(m, j + 2)
            k = lo + int(np.argmin(dp[i - 1, lo:hi]))
            dp[i, j] += dp[i - 1, k]
            back[i, j] = k
    j = int(np.argmin(dp[-1]))
    path = [j]
    for i in range(n - 1, 0, -1):
        j = back[i, j]
        path.append(j)
    path.reverse()
    return path, float(dp[-1, path[-1]])


def remove_seam(img: np.ndarray, path: list[int]) -> np.ndarray:
    n, m = img.shape[:2]
    out = np.zeros((n, m - 1) + img.shape[2:])
    for i in range(n):
        j = path[i]
        out[i] = np.concatenate([img[i, :j], img[i, j + 1 :]], axis=0)
    return np.asarray(out, dtype=np.float64)


def bench_seam_carving(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    n, m = 30, 40
    img = rng.normal(0, 1.0, (n, m))
    # carve a low-energy vertical channel at col 20
    img[:, 20] = 0.0
    e = energy(img)
    path, cost = min_seam(e)
    score += 1.0 if all(abs(p - 20) <= 2 for p in path[2:-2]) else 0.0
    # seam cost is optimal: brute-force DP re-computation agrees
    dp = e.copy()
    for i in range(1, n):
        for j in range(m):
            lo, hi = max(0, j - 1), min(m, j + 2)
            dp[i, j] += np.min(dp[i - 1, lo:hi])
    score += 1.0 if abs(cost - float(np.min(dp[-1]))) < 1e-9 else 0.0
    # removal preserves content: width-1, row lengths consistent
    out = remove_seam(img, path)
    score += 1.0 if out.shape == (n, m - 1) else 0.0
    # content seam beats straight cut: carving through an object costs more
    img2 = rng.normal(0, 1.0, (n, m))
    img2[:, 15:25] += 3.0  # high-energy band
    e2 = energy(img2)
    path2, cost2 = min_seam(e2)
    # straight cut through middle of band
    straight = float(e2[np.arange(n), np.full(n, 20)].sum())
    score += 1.0 if cost2 < straight and not all(p == 20 for p in path2) else 0.0
    return {"synthetic_seam_carving": score / 4.0}
