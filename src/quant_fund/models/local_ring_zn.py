"""Localization Z_(p): elements a/s with p not|s; unique maximal ideal (SYNTHETIC)."""

from __future__ import annotations

from fractions import Fraction


def is_unit_local(f: Fraction, p: int) -> bool:
    """a/s (s coprime p) is unit iff p does not divide a."""
    return f.numerator % p != 0


def localize(a: int, s: int, p: int) -> Fraction:
    if s % p == 0:
        raise ValueError("s not in multiplicative set")
    return Fraction(a, s)


def maximal_ideal_member(f: Fraction, p: int) -> bool:
    """Member of maximal ideal p*Z_(p): numerator divisible by p."""
    return f.numerator % p == 0


def _bench_local_ring_zn(seed: int = 0) -> float:
    checks = []
    p = 3
    x = localize(6, 5, p)  # in maximal ideal
    y = localize(7, 4, p)  # unit
    checks.append(maximal_ideal_member(x, p))
    checks.append(not maximal_ideal_member(y, p))
    checks.append(is_unit_local(y, p))
    checks.append(not is_unit_local(x, p))
    # local ring: non-units form an ideal (sum of non-units is non-unit)
    z = localize(9, 7, p)
    checks.append(maximal_ideal_member(x + z, p))
    # inverse of unit stays in ring
    checks.append(is_unit_local(1 / y, p))
    return float(sum(checks) / len(checks))


def bench_local_ring_zn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_local_ring_zn": _bench_local_ring_zn(seed)}
