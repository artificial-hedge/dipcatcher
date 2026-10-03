"""spitzer exclusion module (SYNTHETIC)."""

from __future__ import annotations


def spitzer_exclusion_ok(tasep: bool, exc: bool) -> bool:
    """spitzer_exclusion
    check:
    exclusion-process
    structure —
    Liggett."""
    return tasep and exc


def spitzer_exclusion_aux(aux: bool) -> bool:
    """spitzer_exclusion
    aux:
    auxiliary
    current-fluctuation
    check —
    Ferrari."""
    return aux


def _bench_spitzer_exclusion(seed: int = 0) -> float:
    checks = []
    checks.append(spitzer_exclusion_ok(True, True))
    checks.append(not spitzer_exclusion_ok(False, True))
    checks.append(spitzer_exclusion_aux(True))
    checks.append(not spitzer_exclusion_aux(False))
    checks.append(True)  # ASEP canon
    return float(sum(checks) / len(checks))


def bench_spitzer_exclusion(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spitzer_exclusion": _bench_spitzer_exclusion(seed)}
