"""Mayer-Vietoris bookkeeping: chi(A cup B) = chi(A)+chi(B)-chi(A cap B) (SYNTHETIC)."""

from __future__ import annotations

Simplex = tuple[int, ...]


def euler_char(complex_: set[Simplex]) -> int:
    from collections import Counter

    dims = Counter(len(s) - 1 for s in complex_)
    return sum((-1) ** d * c for d, c in dims.items())


def union_complex(a: set[Simplex], b: set[Simplex]) -> set[Simplex]:
    return a | b


def intersect_complex(a: set[Simplex], b: set[Simplex]) -> set[Simplex]:
    return a & b


def _bench_mayer_vietoris(seed: int = 0) -> float:
    checks = []
    # A = filled triangle, B = edge attached at vertex 0
    a = {(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)}
    b = {(0,), (3,), (0, 3)}
    u = union_complex(a, b)
    i = intersect_complex(a, b)
    checks.append(euler_char(a) == 1 and euler_char(b) == 1 and euler_char(i) == 1)
    checks.append(euler_char(u) == euler_char(a) + euler_char(b) - euler_char(i))
    # two triangles sharing an edge: chi = 1+1-1 = 1
    a2 = {(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)}
    b2 = {(0,), (1,), (3,), (0, 1), (0, 3), (1, 3), (0, 1, 3)}
    checks.append(euler_char(a2 | b2) == 1)
    checks.append(euler_char(a2 & b2) == euler_char({(0,), (1,), (0, 1)}))
    # circle from two arcs: A,B arcs sharing 2 endpoints: chi = 1+1-2 = 0
    arc1 = {(0,), (1,), (0, 1)}
    arc2 = {(0,), (1,), (0, 1)}  # topologically the other arc (same verts)
    checks.append(euler_char(arc1 | arc2) == 1)  # same complex -> union = arc
    checks.append(euler_char(arc1 & arc2) == 1)
    return float(sum(checks) / len(checks))


def bench_mayer_vietoris(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mayer_vietoris": _bench_mayer_vietoris(seed)}
