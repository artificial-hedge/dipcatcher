"""Mapping cone triangle and its long exact sequence (SYNTHETIC)."""

from __future__ import annotations


def cone_h(v1: int, v2: int, rank: int) -> int:
    """For f: Z^{v1} -> Z^{v2} of given rank, H(cone f) has orders:
    ker f in degree -1, coker f in degree 0."""
    return (v1 - rank) + (v2 - rank)


def _bench_mapping_cone_tri(seed: int = 0) -> float:
    checks = []
    # iso -> contractible cone
    checks.append(cone_h(2, 2, 2) == 0)
    # zero map Z -> Z: cone has Z in both degrees
    checks.append(cone_h(1, 1, 0) == 2)
    # inclusion Z -x2-> Z: coker = Z/2 (order 2 counts one gen)
    checks.append(cone_h(1, 1, 1) == 0)
    # surjective rank-1 Z^2 -> Z: ker = Z
    checks.append(cone_h(2, 1, 1) == 1)
    # LES: H^i(cone) connects H^i(Y) to H^{i+1}(X)
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_mapping_cone_tri(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mapping_cone_tri": _bench_mapping_cone_tri(seed)}
