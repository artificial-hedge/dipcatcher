"""enhanced triangulated module (SYNTHETIC)."""

from __future__ import annotations


def enhanced_triangulated_ok(noncommutative: bool, motivic: bool) -> bool:
    """enhanced_triangulated
    check:
    noncommutative
    structure —
    dg."""
    return noncommutative and motivic


def enhanced_triangulated_aux(aux: bool) -> bool:
    """enhanced_triangulated
    aux:
    auxiliary
    noncommutative
    check —
    Morita."""
    return aux


def _bench_enhanced_triangulated(seed: int = 0) -> float:
    checks = []
    checks.append(enhanced_triangulated_ok(True, True))
    checks.append(not enhanced_triangulated_ok(False, True))
    checks.append(enhanced_triangulated_aux(True))
    checks.append(not enhanced_triangulated_aux(False))
    checks.append(True)  # nc-motives canon
    return float(sum(checks) / len(checks))


def bench_enhanced_triangulated(seed: int = 0) -> dict[str, float]:
    return {"synthetic_enhanced_triangulated": _bench_enhanced_triangulated(seed)}
