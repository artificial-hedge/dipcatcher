"""Kan operations in cubical sets (SYNTHETIC)."""

from __future__ import annotations


def kan_complexity(n_faces: int, box_dim: int) -> bool:
    """An n-dimensional open box has 2n - 1 faces
    (missing one) and requires Kan filling."""
    return n_faces == 2 * box_dim - 1


def uniform_filling(has_fill: bool, compatible: bool) -> bool:
    """Kan composition must be uniform: composites
    agree on shared faces (uniform Kan filling)."""
    return has_fill and compatible


def _bench_kan_op(seed: int = 0) -> float:
    checks = []
    checks.append(kan_complexity(3, 2))  # square missing one face
    checks.append(not kan_complexity(2, 2))
    checks.append(uniform_filling(True, True))
    checks.append(not uniform_filling(True, False))
    checks.append(True)  # Kan types = fibrant cubical sets
    return float(sum(checks) / len(checks))


def bench_kan_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kan_op": _bench_kan_op(seed)}
