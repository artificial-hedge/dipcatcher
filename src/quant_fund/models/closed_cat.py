"""Cartesian closed structure of Set: currying (SYNTHETIC)."""

from __future__ import annotations


def hom_count(x: int, y: int) -> int:
    """|Hom(X, Y)| = |Y|^|X|."""
    return int(y**x)


def exp_obj(y: int, z: int) -> int:
    """|Z^Y| = |Z|^|Y|."""
    return int(z**y)


def curry_check(x: int, y: int, z: int) -> bool:
    """Hom(X x Y, Z) ~ Hom(X, Z^Y): cardinalities agree."""
    return hom_count(x * y, z) == hom_count(x, exp_obj(y, z))


def _bench_closed_cat(seed: int = 0) -> float:
    checks = []
    # |Hom(3-elt, 2-elt)| = 8
    checks.append(hom_count(3, 2) == 8)
    # curry cardinalities: 2^(3*2) = (2^2)^3 = 64
    checks.append(curry_check(3, 2, 2))
    checks.append(curry_check(2, 3, 4))
    # exponential law Z^(X x Y) ~ (Z^Y)^X
    checks.append(exp_obj(6, 2) == 2**6 == 64)
    # X^0 = 1 and X^1 = X
    checks.append(hom_count(0, 5) == 1)
    checks.append(hom_count(1, 5) == 5)
    return float(sum(checks) / len(checks))


def bench_closed_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_closed_cat": _bench_closed_cat(seed)}
