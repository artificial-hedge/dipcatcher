"""liggett exclusion module (SYNTHETIC)."""

from __future__ import annotations


def liggett_exclusion_ok(tasep: bool, exc: bool) -> bool:
    """liggett_exclusion
    check:
    exclusion-process
    structure —
    Liggett."""
    return tasep and exc


def liggett_exclusion_aux(aux: bool) -> bool:
    """liggett_exclusion
    aux:
    auxiliary
    current-fluctuation
    check —
    Ferrari."""
    return aux


def _bench_liggett_exclusion(seed: int = 0) -> float:
    checks = []
    checks.append(liggett_exclusion_ok(True, True))
    checks.append(not liggett_exclusion_ok(False, True))
    checks.append(liggett_exclusion_aux(True))
    checks.append(not liggett_exclusion_aux(False))
    checks.append(True)  # ASEP canon
    return float(sum(checks) / len(checks))


def bench_liggett_exclusion(seed: int = 0) -> dict[str, float]:
    return {"synthetic_liggett_exclusion": _bench_liggett_exclusion(seed)}
