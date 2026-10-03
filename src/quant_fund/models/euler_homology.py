"""Euler characteristic equals alternating sum of Betti numbers (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.simplicial_homology import betti


def euler_count(counts: list[int]) -> int:
    return int(sum((-1) ** i * c for i, c in enumerate(counts)))


def euler_betti(b: tuple[int, ...]) -> int:
    return int(sum((-1) ** i * bi for i, bi in enumerate(b)))


def _bench_euler_homology(seed: int = 0) -> float:
    checks = []
    # tetrahedron: V-E+F = 4-6+4 = 2 = b0-b1+b2
    v = [0, 1, 2, 3]
    e = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    f = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    checks.append(euler_count([4, 6, 4]) == 2)
    checks.append(euler_betti(betti(v, e, f)) == 2)
    # hollow triangle: 3-3 = 0 = 1-1
    checks.append(euler_count([3, 3]) == 0)
    checks.append(euler_betti(betti([0, 1, 2], [(0, 1), (1, 2), (0, 2)], [])) == 0)
    # figure-eight wedge: chi = 3 - 4 = -1
    checks.append(euler_count([3, 4]) == -1)
    # contractible point: chi = 1
    checks.append(euler_betti(betti([0], [], [])) == 1)
    return float(sum(checks) / len(checks))


def bench_euler_homology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_euler_homology": _bench_euler_homology(seed)}
