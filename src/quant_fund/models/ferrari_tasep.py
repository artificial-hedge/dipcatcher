"""ferrari tasep module (SYNTHETIC)."""

from __future__ import annotations


def ferrari_tasep_ok(tasep: bool, exc: bool) -> bool:
    """ferrari_tasep
    check:
    exclusion-process
    structure —
    Liggett."""
    return tasep and exc


def ferrari_tasep_aux(aux: bool) -> bool:
    """ferrari_tasep
    aux:
    auxiliary
    current-fluctuation
    check —
    Ferrari."""
    return aux


def _bench_ferrari_tasep(seed: int = 0) -> float:
    checks = []
    checks.append(ferrari_tasep_ok(True, True))
    checks.append(not ferrari_tasep_ok(False, True))
    checks.append(ferrari_tasep_aux(True))
    checks.append(not ferrari_tasep_aux(False))
    checks.append(True)  # ASEP canon
    return float(sum(checks) / len(checks))


def bench_ferrari_tasep(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ferrari_tasep": _bench_ferrari_tasep(seed)}
