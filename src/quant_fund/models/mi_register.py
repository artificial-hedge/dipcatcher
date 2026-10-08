"""Mattes mutual-information rigid registration (SYNTHETIC).

Joint histogram over intensity bins under a candidate translation; MI =
H(A) + H(B) - H(A,B). A coarse grid search maximizes MI; bench recovers a
known integer shift and beats zero-shift MI.
"""

import numpy as np

_SEED = 20261231 + 877


def _joint_hist(a: np.ndarray, b: np.ndarray, bins: int = 32) -> np.ndarray:
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max()) + 1e-9
    ia = np.clip(((a - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
    ib = np.clip(((b - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
    h = np.zeros((bins, bins))
    np.add.at(h, (ia.ravel(), ib.ravel()), 1)
    return h / h.sum()


def mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    h = _joint_hist(a, b)
    pa, pb = h.sum(axis=1), h.sum(axis=0)

    def _ent(p: np.ndarray) -> float:
        return -float((p[p > 0] * np.log(p[p > 0])).sum())

    return _ent(pa) + _ent(pb) - _ent(h.ravel())


def _shift(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


def register(fixed: np.ndarray, moving: np.ndarray, radius: int = 8) -> tuple[int, int, float]:
    best = (-1, -1, -np.inf)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            mi = mutual_information(fixed, _shift(moving, dy, dx))
            if mi > best[2]:
                best = (dy, dx, mi)
    return best


def bench_mi_register(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: recovers a known shift; aligned MI beats zero-shift MI."""
    rng = np.random.default_rng(seed)
    n = 40
    yy, xx = np.mgrid[0:n, 0:n] / n
    fixed = ((xx - 0.5) ** 2 + (yy - 0.45) ** 2 <= 0.1).astype(float)
    fixed += ((xx - 0.3) ** 2 + (yy - 0.6) ** 2 <= 0.03).astype(float) * 0.5
    fixed += 0.02 * rng.standard_normal((n, n))
    dy, dx = int(rng.integers(1, 6)), int(rng.integers(1, 6))
    moving = _shift(fixed, -dy, -dx)  # fixed is moving shifted by (dy, dx)
    ry, rx, mi = register(fixed, moving, radius=8)
    mi0 = mutual_information(fixed, moving)
    ok = (ry, rx) == (dy, dx) and mi > mi0
    return {"synthetic_mi_register": 1.0 if ok else 0.0}
