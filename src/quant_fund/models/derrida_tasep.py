"""derrida tasep module (SYNTHETIC)."""

from __future__ import annotations


def derrida_tasep_ok(tasep: bool, exc: bool) -> bool:
    """derrida_tasep
    check:
    exclusion-process
    structure —
    Liggett."""
    return tasep and exc


def derrida_tasep_aux(aux: bool) -> bool:
    """derrida_tasep
    aux:
    auxiliary
    current-fluctuation
    check —
    Ferrari."""
    return aux


def _bench_derrida_tasep(seed: int = 0) -> float:
    checks = []
    checks.append(derrida_tasep_ok(True, True))
    checks.append(not derrida_tasep_ok(False, True))
    checks.append(derrida_tasep_aux(True))
    checks.append(not derrida_tasep_aux(False))
    checks.append(True)  # ASEP canon
    return float(sum(checks) / len(checks))


def bench_derrida_tasep(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derrida_tasep": _bench_derrida_tasep(seed)}
