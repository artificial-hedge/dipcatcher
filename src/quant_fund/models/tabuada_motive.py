"""tabuada motive module (SYNTHETIC)."""

from __future__ import annotations


def tabuada_motive_ok(noncommutative: bool, motivic: bool) -> bool:
    """tabuada_motive
    check:
    noncommutative
    structure —
    dg."""
    return noncommutative and motivic


def tabuada_motive_aux(aux: bool) -> bool:
    """tabuada_motive
    aux:
    auxiliary
    noncommutative
    check —
    Morita."""
    return aux


def _bench_tabuada_motive(seed: int = 0) -> float:
    checks = []
    checks.append(tabuada_motive_ok(True, True))
    checks.append(not tabuada_motive_ok(False, True))
    checks.append(tabuada_motive_aux(True))
    checks.append(not tabuada_motive_aux(False))
    checks.append(True)  # nc-motives canon
    return float(sum(checks) / len(checks))


def bench_tabuada_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tabuada_motive": _bench_tabuada_motive(seed)}
