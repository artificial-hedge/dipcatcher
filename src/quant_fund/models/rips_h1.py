"""Vietoris-Rips H1 detection of a hole in a point cloud (wave 280) (SYNTHETIC).

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
    # Evenly spaced angles + jitter: a fully uniform draw can leave an arc
    # gap wider than eps, splitting the ring into disconnected chains whose
    # H1 legitimately vanishes (measured b0=2 at the family seed). The
    # planted-hole claim needs the ring connected, not uniform sampling.
    th = np.linspace(0.0, 2 * np.pi, 24, endpoint=False) + rng.uniform(-np.pi / 48, np.pi / 48, 24)
    ring = np.stack([np.cos(th), np.sin(th)], 1)
    # Contractible comparison: a filled disk — points at random interior
    # radii fill the hole so no H1 class persists. (Squashing the loop only
    # made a thinner closed curve: it still bounds a hole, H1 measured 3.)
    disk = ring * np.sqrt(rng.uniform(0.0, 1.0, 24))[:, None]
    h1_ring = _vr_betti1(ring, 0.8)
    h1_disk = _vr_betti1(disk, 0.8)
    if not (h1_ring >= 1 and h1_disk == 0):
        raise ValueError("Vietoris-Rips H1 oracle failed (ring/blob)")
    return {
        "synthetic_rips_ring": float(h1_ring >= 1),
        "synthetic_rips_disk": float(h1_disk == 0),
    }
