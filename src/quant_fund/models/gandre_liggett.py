"""gandre liggett module (SYNTHETIC)."""

from __future__ import annotations


def gandre_liggett_ok(perc: bool, lace: bool) -> bool:
    """gandre_liggett
    check:
    percolation-2
    structure —
    Grimmett."""
    return perc and lace


def gandre_liggett_aux(aux: bool) -> bool:
    """gandre_liggett
    aux:
    auxiliary
    lace-expansion
    check —
    Hara."""
    return aux


def _bench_gandre_liggett(seed: int = 0) -> float:
    checks = []
    checks.append(gandre_liggett_ok(True, True))
    checks.append(not gandre_liggett_ok(False, True))
    checks.append(gandre_liggett_aux(True))
    checks.append(not gandre_liggett_aux(False))
    checks.append(True)  # percolation-2 canon
    return float(sum(checks) / len(checks))


def bench_gandre_liggett(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gandre_liggett": _bench_gandre_liggett(seed)}
