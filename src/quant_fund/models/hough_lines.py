"""Hough transform line detection: (rho, theta) accumulator + peak picking.

SYNTHETIC bench only.
"""

import numpy as np

_SEED = 20261231 + 909


def hough_accumulate(
    edge: np.ndarray, ntheta: int = 180
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge = np.asarray(edge, dtype=np.float64)
    ys, xs = np.nonzero(edge > 0.5)
    thetas = np.linspace(-np.pi / 2, np.pi / 2, ntheta)
    diag = int(np.hypot(*edge.shape))
    rhos = np.linspace(-diag, diag, 2 * diag + 1)
    acc = np.zeros((rhos.size, ntheta))
    cs, sn = np.cos(thetas), np.sin(thetas)
    for y, x in zip(ys, xs, strict=True):
        r = x * cs + y * sn
        idx = np.clip(np.round(r + diag).astype(int), 0, rhos.size - 1)
        acc[idx, np.arange(ntheta)] += 1.0
    return (
        np.asarray(acc, dtype=np.float64),
        np.asarray(rhos, dtype=np.float64),
        np.asarray(thetas, dtype=np.float64),
    )


def hough_peaks(acc: np.ndarray, k: int = 4, nms: int = 5) -> list[tuple[int, int]]:
    work = acc.copy()
    out: list[tuple[int, int]] = []
    for _ in range(k):
        i, j = np.unravel_index(np.argmax(work), work.shape)
        if work[i, j] <= 0:
            break
        out.append((int(i), int(j)))
        lo0, hi0 = max(0, i - nms), min(work.shape[0], i + nms + 1)
        lo1, hi1 = max(0, j - nms), min(work.shape[1], j + nms + 1)
        work[lo0:hi0, lo1:hi1] = 0.0
    return out


def bench_hough_lines(seed: int = _SEED) -> dict[str, float]:
    score = 0.0
    n = 60
    edge = np.zeros((n, n))
    # horizontal line y=30 and vertical line x=40
    edge[30, 5:55] = 1.0
    edge[5:55, 40] = 1.0
    acc, rhos, thetas = hough_accumulate(edge)
    peaks = hough_peaks(acc, k=4, nms=6)
    found_h = any(
        abs(abs(rhos[p[0]]) - 30.0) < 2.0 and abs(abs(thetas[p[1]]) - np.pi / 2) < 0.06
        for p in peaks
    )
    found_v = any(abs(rhos[p[0]] - 40.0) < 2.0 and abs(thetas[p[1]]) < 0.06 for p in peaks)
    score += 1.0 if found_h else 0.0
    score += 1.0 if found_v else 0.0
    # diagonal line y=x: rho=0, theta=-45°
    edge2 = np.zeros((n, n))
    for i in range(5, 55):
        edge2[i, i] = 1.0
    acc2, rhos2, thetas2 = hough_accumulate(edge2)
    p2 = hough_peaks(acc2, k=1, nms=6)[0]
    score += 1.0 if abs(rhos2[p2[0]]) < 2.0 and abs(abs(thetas2[p2[1]]) - np.pi / 4) < 0.06 else 0.0
    # accumulator count at the true bin ≈ line length
    ih = np.argmin(np.abs(rhos + 30.0))
    jt = np.argmin(np.abs(thetas + np.pi / 2))
    score += 1.0 if acc[ih, jt] >= 45.0 else 0.0
    return {"synthetic_hough_lines": score / 4.0}
