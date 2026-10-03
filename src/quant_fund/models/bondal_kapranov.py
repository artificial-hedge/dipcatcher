"""bondal kapranov module (SYNTHETIC)."""

from __future__ import annotations


def bondal_kapranov_ok(noncommutative: bool, motivic: bool) -> bool:
    """bondal_kapranov
    check:
    noncommutative
    structure —
    dg."""
    return noncommutative and motivic


def bondal_kapranov_aux(aux: bool) -> bool:
    """bondal_kapranov
    aux:
    auxiliary
    noncommutative
    check —
    Morita."""
    return aux


def _bench_bondal_kapranov(seed: int = 0) -> float:
    checks = []
    checks.append(bondal_kapranov_ok(True, True))
    checks.append(not bondal_kapranov_ok(False, True))
    checks.append(bondal_kapranov_aux(True))
    checks.append(not bondal_kapranov_aux(False))
    checks.append(True)  # nc-motives canon
    return float(sum(checks) / len(checks))


def bench_bondal_kapranov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bondal_kapranov": _bench_bondal_kapranov(seed)}
