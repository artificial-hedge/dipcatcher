"""sasamoto tasep module (SYNTHETIC)."""

from __future__ import annotations


def sasamoto_tasep_ok(tasep: bool, exc: bool) -> bool:
    """sasamoto_tasep
    check:
    exclusion-process
    structure —
    Liggett."""
    return tasep and exc


def sasamoto_tasep_aux(aux: bool) -> bool:
    """sasamoto_tasep
    aux:
    auxiliary
    current-fluctuation
    check —
    Ferrari."""
    return aux


def _bench_sasamoto_tasep(seed: int = 0) -> float:
    checks = []
    checks.append(sasamoto_tasep_ok(True, True))
    checks.append(not sasamoto_tasep_ok(False, True))
    checks.append(sasamoto_tasep_aux(True))
    checks.append(not sasamoto_tasep_aux(False))
    checks.append(True)  # ASEP canon
    return float(sum(checks) / len(checks))


def bench_sasamoto_tasep(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sasamoto_tasep": _bench_sasamoto_tasep(seed)}
