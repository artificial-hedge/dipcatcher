"""Vietoris-Rips H1 detection of a hole in a point cloud (wave 280).

Ring-shaped samples produce an H1 class at a band of epsilon; a filled disk
at the same epsilon has none. H1 is computed via the flag complex Betti-1.
"""

import numpy as np

from quant_fund.models.simp_betti import betti

_SEED = 20261231 + 767


def _vr_betti1(pts: np.ndarray, eps: float) -> int:
    n = len(pts)
    d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    edges: list[tuple[int, ...]] = [
        (i, j) for i in range(n) for j in range(i + 1, n) if d[i, j] < eps
    ]
    tris: list[tuple[int, ...]] = [
        (i, j, k) for (i, j) in edges for k in range(j + 1, n) if d[i, k] < eps and d[j, k] < eps
    ]
    comp: list[list[tuple[int, ...]]] = [[(i,) for i in range(n)], edges, tris]
    b = betti(comp)
    return b[1] if len(b) > 1 else 0


def bench_rips_h1(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    th = rng.uniform(0, 2 * np.pi, 24)
    ring = np.stack([np.cos(th), np.sin(th)], 1)
    disk = ring.copy()
    disk[:, 0] *= 0.4  # squash hole -> contractible blob (still sparse center)
    disk[:, 1] *= 3.0
    h1_ring = _vr_betti1(ring, 0.8)
    h1_disk = _vr_betti1(disk, 0.8)
    return {
        "synthetic_rips_ring": float(h1_ring >= 1),
        "synthetic_rips_disk": float(h1_disk == 0),
    }
