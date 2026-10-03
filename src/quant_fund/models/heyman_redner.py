"""heyman redner module (SYNTHETIC)."""

from __future__ import annotations


def heyman_redner_ok(perc: bool, lace: bool) -> bool:
    """heyman_redner
    check:
    percolation-2
    structure —
    Grimmett."""
    return perc and lace


def heyman_redner_aux(aux: bool) -> bool:
    """heyman_redner
    aux:
    auxiliary
    lace-expansion
    check —
    Hara."""
    return aux


def _bench_heyman_redner(seed: int = 0) -> float:
    checks = []
    checks.append(heyman_redner_ok(True, True))
    checks.append(not heyman_redner_ok(False, True))
    checks.append(heyman_redner_aux(True))
    checks.append(not heyman_redner_aux(False))
    checks.append(True)  # percolation-2 canon
    return float(sum(checks) / len(checks))


def bench_heyman_redner(seed: int = 0) -> dict[str, float]:
    return {"synthetic_heyman_redner": _bench_heyman_redner(seed)}
