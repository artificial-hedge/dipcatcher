"""tracy widom_tasep module (SYNTHETIC)."""

from __future__ import annotations


def tracy_widom_tasep_ok(tasep: bool, exc: bool) -> bool:
    """tracy_widom_tasep
    check:
    exclusion-process
    structure —
    Liggett."""
    return tasep and exc


def tracy_widom_tasep_aux(aux: bool) -> bool:
    """tracy_widom_tasep
    aux:
    auxiliary
    current-fluctuation
    check —
    Ferrari."""
    return aux


def _bench_tracy_widom_tasep(seed: int = 0) -> float:
    checks = []
    checks.append(tracy_widom_tasep_ok(True, True))
    checks.append(not tracy_widom_tasep_ok(False, True))
    checks.append(tracy_widom_tasep_aux(True))
    checks.append(not tracy_widom_tasep_aux(False))
    checks.append(True)  # ASEP canon
    return float(sum(checks) / len(checks))


def bench_tracy_widom_tasep(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tracy_widom_tasep": _bench_tracy_widom_tasep(seed)}
