"""Hu moment invariants — 7 rotation/scale/translation-invariant image moments (SYNTHETIC).

Central moments eta_pq normalized by m00^(1+(p+q)/2); the seven Hu
invariants are polynomial combinations. Bench: invariants match between
a shape and its rotated/scaled/translated copy.
"""

import numpy as np

_SEED = 20261231 + 875


def _raw_moments(img: np.ndarray, p: int, q: int) -> float:
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    return float((xx**p * yy**q * img).sum())


def _central(img: np.ndarray, p: int, q: int) -> float:
    m00 = _raw_moments(img, 0, 0)
    xc = _raw_moments(img, 1, 0) / m00
    yc = _raw_moments(img, 0, 1) / m00
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    return float(float(((xx - xc) ** p * (yy - yc) ** q * img).sum()))


def hu_moments(img: np.ndarray) -> np.ndarray:
    m00 = _raw_moments(img, 0, 0)

    def eta(p: int, q: int) -> float:
        return float(_central(img, p, q) / m00 ** (1 + (p + q) / 2))

    n20, n02, n11 = eta(2, 0), eta(0, 2), eta(1, 1)
    n30, n12, n21, n03 = eta(3, 0), eta(1, 2), eta(2, 1), eta(0, 3)
    return np.array(
        [
            n20 + n02,
            (n20 - n02) ** 2 + 4 * n11**2,
            (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2,
            (n30 + n12) ** 2 + (n21 + n03) ** 2,
            (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
            + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
            (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2)
            + 4 * n11 * (n30 + n12) * (n21 + n03),
            (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
            - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
        ]
    )


def _transform(img: np.ndarray, scale: float, theta: float, tx: float, ty: float) -> np.ndarray:
    n = img.shape[0]
    c = (n - 1) / 2
    co, si = np.cos(theta), np.sin(theta)
    yy, xx = np.mgrid[0:n, 0:n]
    xs = (co * (xx - c - tx) + si * (yy - c - ty)) / scale + c
    ys = (-si * (xx - c - tx) + co * (yy - c - ty)) / scale + c
    xi, yi = np.round(xs).astype(int), np.round(ys).astype(int)
    valid = (xi >= 0) & (xi < n) & (yi >= 0) & (yi < n)
    out = np.zeros_like(img)
    out[valid] = img[yi[valid], xi[valid]]
    return out


def bench_hu_moments(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: invariants match across rigid transform; differ across shapes."""
    n = 48
    yy, xx = np.mgrid[0:n, 0:n] / n - 0.5
    img = np.zeros((n, n))
    img[np.abs(xx / 0.3) + np.abs(yy / 0.2) <= 1] = 1.0  # diamond
    t = _transform(img, 1.3, 0.7, 3.0, -4.0)
    h1, h2 = hu_moments(img), hu_moments(t)
    rel = np.abs(np.log(np.abs(h1) + 1e-12) - np.log(np.abs(h2) + 1e-12))
    other = ((xx / 0.25) ** 2 + (yy / 0.25) ** 2 <= 1).astype(float)  # circle
    h3 = hu_moments(other)
    diff = np.abs(np.log(np.abs(h1) + 1e-12) - np.log(np.abs(h3) + 1e-12))
    ok = rel[0] < 0.08 and diff[0] > 0.1
    return {"synthetic_hu_moments": 1.0 if ok else 0.0}
