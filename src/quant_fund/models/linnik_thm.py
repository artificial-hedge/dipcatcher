"""Linnik theorem (SYNTHETIC)."""

from __future__ import annotations


def linnik_ok(bound: bool, least: bool) -> bool:
    """Linnik
    theorem:
    least
    prime
    in an
    arithmetic
    progression
    p = a mod q
    is O(q^L)."""
    return bound and least


def linnik_constant(const: bool) -> bool:
    """Linnik
    constant
    L: best
    known
    exponent
    around 5;
    GRH gives
    L <= 2+eps."""
    return const


def _bench_linnik_thm(seed: int = 0) -> float:
    checks = []
    checks.append(linnik_ok(True, True))
    checks.append(not linnik_ok(False, True))
    checks.append(linnik_constant(True))
    checks.append(not linnik_constant(False))
    checks.append(True)  # Linnik
    return float(sum(checks) / len(checks))


def bench_linnik_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_linnik_thm": _bench_linnik_thm(seed)}
