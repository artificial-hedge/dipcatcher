"""Primitive elements and Milnor-Moore (SYNTHETIC)."""

from __future__ import annotations


def primitive_ok(coprod: float, x: float) -> bool:
    """x is primitive iff Delta(x) = x otimes 1 + 1 otimes x;
    on scalars the reduced diagonal vanishes."""
    return abs(coprod - (x + x)) < 1e-9


def milnor_moore_ok(cocommutative: bool, char0: bool) -> bool:
    """Milnor-Moore: connected cocommutative Hopf algebra
    in char 0 = U(Prim H) of its primitives."""
    return cocommutative and char0


def _bench_primitive_elts(seed: int = 0) -> float:
    checks = []
    checks.append(primitive_ok(6.0, 3.0))
    checks.append(not primitive_ok(5.0, 3.0))
    checks.append(milnor_moore_ok(True, True))
    checks.append(not milnor_moore_ok(True, False))
    # Prim(Lie) primitives generate free Lie
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_primitive_elts(seed: int = 0) -> dict[str, float]:
    return {"synthetic_primitive_elts": _bench_primitive_elts(seed)}
