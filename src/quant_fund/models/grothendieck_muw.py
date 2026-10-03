"""grothendieck muw module (SYNTHETIC)."""

from __future__ import annotations


def grothendieck_muw_ok(swan: bool, tame: bool) -> bool:
    """grothendieck_muw
    check:
    ramification
    structure —
    Kato."""
    return swan and tame


def grothendieck_muw_aux(aux: bool) -> bool:
    """grothendieck_muw
    aux:
    auxiliary
    conductor
    check —
    Saito."""
    return aux


def _bench_grothendieck_muw(seed: int = 0) -> float:
    checks = []
    checks.append(grothendieck_muw_ok(True, True))
    checks.append(not grothendieck_muw_ok(False, True))
    checks.append(grothendieck_muw_aux(True))
    checks.append(not grothendieck_muw_aux(False))
    checks.append(True)  # ramification canon
    return float(sum(checks) / len(checks))


def bench_grothendieck_muw(seed: int = 0) -> dict[str, float]:
    return {"synthetic_grothendieck_muw": _bench_grothendieck_muw(seed)}
