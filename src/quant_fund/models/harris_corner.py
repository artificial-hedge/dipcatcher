"""Harris corner detection: structure tensor, corner response, NMS.

SYNTHETIC bench only.
"""

import numpy as np

_SEED = 20261231 + 908


def _grad(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    return np.asarray(gx * 0.5, dtype=np.float64), np.asarray(gy * 0.5, dtype=np.float64)


def _box(x: np.ndarray, k: int = 3) -> np.ndarray:
    out = np.zeros_like(x)
    r = k // 2
    cs = np.cumsum(np.cumsum(np.pad(x, r + 1), axis=0), axis=1)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            i0, j0 = i, j
            i1, j1 = i + 2 * r + 1, j + 2 * r + 1
            out[i, j] = cs[i1, j1] - cs[i0, j1] - cs[i1, j0] + cs[i0, j0]
    return np.asarray(out, dtype=np.float64)


def harris_response(img: np.ndarray, k: float = 0.04) -> np.ndarray:
    img = np.asarray(img, dtype=np.float64)
    gx, gy = _grad(img)
    sxx, sxy, syy = _box(gx * gx), _box(gx * gy), _box(gy * gy)
    det = sxx * syy - sxy * sxy
    tr = sxx + syy
    return np.asarray(det - k * tr * tr, dtype=np.float64)


def detect_corners(r: np.ndarray, thresh: float, nms: int = 3) -> list[tuple[int, int]]:
    r = np.asarray(r, dtype=np.float64)
    pts: list[tuple[int, int]] = []
    rmax = np.max(r)
    work = r.copy()
    while True:
        i, j = np.unravel_index(np.argmax(work), work.shape)
        if work[i, j] < thresh:
            break
        pts.append((int(i), int(j)))
        lo0, hi0 = max(0, i - nms), min(work.shape[0], i + nms + 1)
        lo1, hi1 = max(0, j - nms), min(work.shape[1], j + nms + 1)
        work[lo0:hi0, lo1:hi1] = -1e18
    _ = rmax
    return pts


def bench_harris_corner(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    n = 60
    img = rng.normal(0, 0.02, (n, n))
    # checkerboard block → corners at its 4 edges
    img[20:40, 20:40] += 1.0
    r = harris_response(img)
    pts = detect_corners(r, thresh=0.3 * np.max(r), nms=4)
    expected = [(20, 20), (20, 39), (39, 20), (39, 39)]
    hits = sum(any(abs(p[0] - e[0]) <= 2 and abs(p[1] - e[1]) <= 2 for p in pts) for e in expected)
    score += 1.0 if hits >= 3 else 0.0
    # flat region: no corners
    img_flat = np.ones((n, n)) * 0.5 + rng.normal(0, 0.001, (n, n))
    pts_f = detect_corners(
        harris_response(img_flat), thresh=0.3 * np.max(harris_response(img)), nms=4
    )
    score += 1.0 if len(pts_f) <= 2 else 0.0
    # edge (not corner): response low relative to corner
    img_e = np.zeros((n, n))
    img_e[20:40, :] = 1.0
    r_e = harris_response(img_e)
    corner_val = np.max(r)
    edge_val = np.max(r_e[25:35, 10:50])
    score += 1.0 if edge_val < 0.05 * corner_val else 0.0
    # shift invariance: translate block 3px → same corners shifted
    img2 = rng.normal(0, 0.02, (n, n))
    img2[23:43, 23:43] += 1.0
    pts2 = detect_corners(harris_response(img2), thresh=0.3 * np.max(harris_response(img2)), nms=4)
    exp2 = [(23, 23), (23, 42), (42, 23), (42, 42)]
    hits2 = sum(any(abs(p[0] - e[0]) <= 2 and abs(p[1] - e[1]) <= 2 for p in pts2) for e in exp2)
    score += 1.0 if hits2 >= 3 else 0.0
    return {"synthetic_harris_corner": score / 4.0}
