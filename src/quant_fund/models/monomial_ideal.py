"""Monomial ideals: membership, minimal generators, lcm (SYNTHETIC)."""

from __future__ import annotations

Mon = tuple[int, ...]  # exponent vector


def divides(m1: Mon, m2: Mon) -> bool:
    """m1 | m2 exponent-wise."""
    return all(a <= b for a, b in zip(m1, m2, strict=True))


def in_ideal(m: Mon, gens: frozenset[Mon]) -> bool:
    return any(divides(g, m) for g in gens)


def minimal_gens(gens: frozenset[Mon]) -> frozenset[Mon]:
    """Drop generators divisible by another."""
    return frozenset(g for g in gens if not any(h != g and divides(h, g) for h in gens))


def lcm_monomials(m1: Mon, m2: Mon) -> Mon:
    return tuple(max(a, b) for a, b in zip(m1, m2, strict=True))


def _bench_monomial_ideal(seed: int = 0) -> float:
    checks = []
    g = frozenset({(2, 0), (1, 1)})  # <x^2, xy>
    checks.append(in_ideal((3, 0), g))
    checks.append(in_ideal((2, 2), g))
    checks.append(not in_ideal((0, 1), g))
    checks.append(not in_ideal((1, 0), g))
    checks.append(minimal_gens(frozenset({(2, 0), (3, 0), (1, 1)})) == frozenset({(2, 0), (1, 1)}))
    checks.append(lcm_monomials((2, 1), (1, 3)) == (2, 3))
    checks.append(divides((1, 0), (2, 1)))
    return float(sum(checks) / len(checks))


def bench_monomial_ideal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_monomial_ideal": _bench_monomial_ideal(seed)}
