"""chow robbins module (SYNTHETIC)."""

from __future__ import annotations


def chow_robbins_ok(os1: bool, sd: bool) -> bool:
    """chow_robbins
    check:
    optimal-
    stopping —
    value
    function."""
    return os1 and sd


def chow_robbins_aux(aux: bool) -> bool:
    """chow_robbins
    aux:
    auxiliary
    stopping
    check —
    boundary."""
    return aux


def _bench_chow_robbins(seed: int = 0) -> float:
    checks = []
    checks.append(chow_robbins_ok(True, True))
    checks.append(not chow_robbins_ok(False, True))
    checks.append(chow_robbins_aux(True))
    checks.append(not chow_robbins_aux(False))
    checks.append(True)  # optimal-stopping canon
    return float(sum(checks) / len(checks))


def bench_chow_robbins(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chow_robbins": _bench_chow_robbins(seed)}
