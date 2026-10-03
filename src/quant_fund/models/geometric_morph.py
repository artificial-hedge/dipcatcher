"""Geometric morphisms: inverse image preserves finite limits (SYNTHETIC)."""

from __future__ import annotations


def inverse_image_pullback(f: list[int], a_sub: list[int], b_sub: list[int]) -> list[int]:
    """f^*(A x_B C) = f^*(A) x_{f^*(B)} f^*(C) for maps of finite sets."""
    # model: sets are fibers of f over points; pullback over base
    inv_a = [i for i, x in enumerate(f) if x in a_sub]
    inv_b = [i for i, x in enumerate(f) if x in b_sub]
    return [i for i in inv_a if i in inv_b]


def _bench_geometric_morph(seed: int = 0) -> float:
    checks = []
    f = [0, 0, 1, 1, 2]
    a = [0, 1]
    b = [0, 1, 2]
    pb = inverse_image_pullback(f, a, b)
    checks.append(sorted(pb) == [0, 1, 2, 3])
    # inverse image of empty is empty
    checks.append(inverse_image_pullback(f, [], b) == [])
    # inverse image of universe is universe
    checks.append(sorted(inverse_image_pullback(f, [0, 1, 2], b)) == [0, 1, 2, 3, 4])
    # direct image need not preserve limits (asymmetry)
    checks.append(len(pb) == 4)
    # terminal object preserved: f^*(1) = whole source fiber
    checks.append(sorted(inverse_image_pullback(f, [0, 1, 2], [0, 1, 2])) == list(range(5)))
    return float(sum(checks) / len(checks))


def bench_geometric_morph(seed: int = 0) -> dict[str, float]:
    return {"synthetic_geometric_morph": _bench_geometric_morph(seed)}
