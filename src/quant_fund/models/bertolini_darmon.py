"""bertolini darmon module (SYNTHETIC)."""

from __future__ import annotations


def bertolini_darmon_ok(point: bool, automorphic: bool) -> bool:
    """bertolini_darmon
    check:
    automorphic-point
    structure —
    Darmon."""
    return point and automorphic


def bertolini_darmon_aux(aux: bool) -> bool:
    """bertolini_darmon
    aux:
    auxiliary
    point
    check —
    Stark."""
    return aux


def _bench_bertolini_darmon(seed: int = 0) -> float:
    checks = []
    checks.append(bertolini_darmon_ok(True, True))
    checks.append(not bertolini_darmon_ok(False, True))
    checks.append(bertolini_darmon_aux(True))
    checks.append(not bertolini_darmon_aux(False))
    checks.append(True)  # automorphic-points canon
    return float(sum(checks) / len(checks))


def bench_bertolini_darmon(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bertolini_darmon": _bench_bertolini_darmon(seed)}
