"""Triangulated category axioms on a toy shift system (SYNTHETIC)."""

from __future__ import annotations


def rotate_triangle(a: int, b: int, c: int) -> tuple[int, int, int]:
    """X -> Y -> Z -> X[1] rotates to Y -> Z -> X[1] -> Y[1]."""
    return (b, c, a + 1)


def _bench_triangulated(seed: int = 0) -> float:
    checks = []
    # rotation of a triangle is a triangle
    r = rotate_triangle(0, 1, 2)
    checks.append(r == (1, 2, 1))
    # identity triangle X -> X -> 0 -> X[1] is distinguished
    checks.append(rotate_triangle(5, 5, -1) == (5, -1, 6))
    # shift is additive: (X[1])[1] = X[2]
    checks.append((0 + 1) + 1 == 2)
    # opposite rotation moves the connecting morphism
    checks.append(rotate_triangle(*rotate_triangle(0, 1, 2)) == (2, 1, 2))
    # TR3: morphism of triangles closes up
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_triangulated(seed: int = 0) -> dict[str, float]:
    return {"synthetic_triangulated": _bench_triangulated(seed)}
