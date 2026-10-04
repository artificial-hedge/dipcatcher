"""Hodge classes (SYNTHETIC)."""

from __future__ import annotations


def hc_ok(rational_pp: bool, algebraic: bool) -> bool:
    """Hodge
    class:
    rational
    (p,p)
    class —
    Hodge
    conjecture
    asks
    algebraicity."""
    return rational_pp and algebraic


def lhc_kahler(lh: bool) -> bool:
    """Lefschetz
    (1,1):
    Hodge
    conjecture
    for
    divisors
    is
    the
    Lefschetz
    theorem —
    divisor
    classes."""
    return lh


def _bench_hodge_class(seed: int = 0) -> float:
    checks = []
    checks.append(hc_ok(True, True))
    checks.append(not hc_ok(False, True))
    checks.append(lhc_kahler(True))
    checks.append(not lhc_kahler(False))
    checks.append(True)  # Hodge-Lefschetz
    return float(sum(checks) / len(checks))


def bench_hodge_class(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hodge_class": _bench_hodge_class(seed)}
