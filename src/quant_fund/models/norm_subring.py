"""Norm in quadratic integer rings Z[sqrt(d)] (SYNTHETIC)."""

from __future__ import annotations


def quad_norm(a: int, b: int, d: int) -> int:
    """N(a + b sqrt(d)) = a^2 - d b^2."""
    return a * a - d * b * b


def quad_mul(x: tuple[int, int], y: tuple[int, int], d: int) -> tuple[int, int]:
    """(a + b sqrt(d))(c + e sqrt(d)) = (ac + dbe) + (ae + bc) sqrt(d)."""
    a, b = x
    c, e = y
    return (a * c + d * b * e, a * e + b * c)


def is_unit(a: int, b: int, d: int) -> bool:
    """Units in Z[sqrt(d)] are exactly the norm +-1 elements."""
    return abs(quad_norm(a, b, d)) == 1


def _bench_norm_subring(seed: int = 0) -> float:
    checks = []
    # N(3 + 2i) = 13 in Z[i]
    checks.append(quad_norm(3, 2, -1) == 13)
    # norm multiplicative: N(xy) = N(x)N(y) on Z[sqrt(2)]
    x, y = (3, 1), (1, 2)
    xy = quad_mul(x, y, 2)
    checks.append(quad_norm(*xy, 2) == quad_norm(*x, 2) * quad_norm(*y, 2))
    # 1 + sqrt(2) is a unit (norm -1)
    checks.append(is_unit(1, 1, 2))
    # 2 + sqrt(2) has norm 2, not a unit
    checks.append(not is_unit(2, 1, 2))
    # -1 has norm 1 in every ring
    checks.append(is_unit(-1, 0, 7))
    # i^2 = -1: (0,1)^2 in Z[i] = (-1,0)
    checks.append(quad_mul((0, 1), (0, 1), -1) == (-1, 0))
    return float(sum(checks) / len(checks))


def bench_norm_subring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_norm_subring": _bench_norm_subring(seed)}
