"""Kronecker-Weber: quadratic fields inside cyclotomic fields (SYNTHETIC)."""

from __future__ import annotations


def quad_disc(d: int) -> int:
    """Discriminant of Q(sqrt d), d squarefree."""
    return d if d % 4 == 1 else 4 * d


def conductor(d: int) -> int:
    """Conductor of Q(sqrt d) = |disc|."""
    return abs(quad_disc(d))


def _bench_kronecker_weber(seed: int = 0) -> float:
    checks = []
    # Q(sqrt -3) sits in Q(zeta_3): disc -3, conductor 3
    checks.append(conductor(-3) == 3)
    # Q(sqrt 5) in Q(zeta_5): disc 5
    checks.append(conductor(5) == 5)
    # Q(sqrt -1) in Q(zeta_4): disc -4, conductor 4
    checks.append(conductor(-1) == 4)
    # Q(sqrt 2) in Q(zeta_8): disc 8
    checks.append(conductor(2) == 8)
    # Q(sqrt -7) in Q(zeta_7): disc -7 (since -7 = 1 mod 4)
    checks.append(conductor(-7) == 7)
    return float(sum(checks) / len(checks))


def bench_kronecker_weber(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kronecker_weber": _bench_kronecker_weber(seed)}
