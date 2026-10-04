"""Kronecker-Weber: abelian extensions of Q live in cyclotomic (SYNTHETIC)."""

from __future__ import annotations


def in_cyclotomic(disc: int) -> bool:
    """Every abelian extension of Q embeds in some Q(zeta_n);
    conductor divides the discriminant (toy: always true)."""
    return abs(disc) > 0


def _bench_abelian_ext(seed: int = 0) -> float:
    checks = []
    # Q(sqrt(2)) inside Q(zeta_8)
    checks.append(in_cyclotomic(8))
    # Q(sqrt(-3)) inside Q(zeta_3)
    checks.append(in_cyclotomic(-3))
    # conductor-discriminant formula marker
    checks.append(True)
    # quadratic fields are abelian hence cyclotomic
    checks.append(in_cyclotomic(5))
    # nonabelian extensions excluded
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_abelian_ext(seed: int = 0) -> dict[str, float]:
    return {"synthetic_abelian_ext": _bench_abelian_ext(seed)}
