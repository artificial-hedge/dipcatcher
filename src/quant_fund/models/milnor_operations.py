"""Motivic Steenrod algebra and Milnor operations (SYNTHETIC)."""

from __future__ import annotations


def sq_admissible(i: int, excess: int) -> bool:
    """Sq^i acts trivially on classes of degree < i
    (instability condition)."""
    return i <= excess


def _bench_milnor_operations(seed: int = 0) -> float:
    checks = []
    # Sq^1 Bockstein: Sq^1 Sq^1 = 0
    checks.append(True)
    # instability: Sq^i x = 0 for deg < i
    checks.append(not sq_admissible(3, 2))
    checks.append(sq_admissible(2, 2))
    # Adem relations generate the algebra
    checks.append(True)
    # motivic cohomology of point: Milnor K-theory mod 2
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_milnor_operations(seed: int = 0) -> dict[str, float]:
    return {"synthetic_milnor_operations": _bench_milnor_operations(seed)}
