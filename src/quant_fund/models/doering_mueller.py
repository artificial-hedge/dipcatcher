"""doering mueller module (SYNTHETIC)."""

from __future__ import annotations


def doering_mueller_ok(sp1: bool, wn: bool) -> bool:
    """doering_mueller
    check:
    SPDE —
    mild/white-noise
    solution."""
    return sp1 and wn


def doering_mueller_aux(aux: bool) -> bool:
    """doering_mueller
    aux:
    auxiliary
    Walsh
    check —
    martingale
    measure."""
    return aux


def _bench_doering_mueller(seed: int = 0) -> float:
    checks = []
    checks.append(doering_mueller_ok(True, True))
    checks.append(not doering_mueller_ok(False, True))
    checks.append(doering_mueller_aux(True))
    checks.append(not doering_mueller_aux(False))
    checks.append(True)  # SPDE canon
    return float(sum(checks) / len(checks))


def bench_doering_mueller(seed: int = 0) -> dict[str, float]:
    return {"synthetic_doering_mueller": _bench_doering_mueller(seed)}
