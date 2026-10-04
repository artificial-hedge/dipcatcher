"""printmaking module (SYNTHETIC)."""

from __future__ import annotations


def printmaking_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """printmaking

    check:
    painting_techniques: painting techniques
    sculpture_methods: sculpture methods
    printmaking: printmaking
    art_conservation: art conservation
    art_history: art history
    visual_culture: visual culture
    """
    return fit_ok and sample_ok


def printmaking_aux(aux: bool) -> bool:
    """printmaking

    aux:
    painting_techniques: pigment and medium
    sculpture_methods: three-dimensional form
    printmaking: impression processes
    art_conservation: preservation methods
    art_history: art historical periods
    visual_culture: visual studies
    """
    return aux


def _bench_printmaking(seed: int = 0) -> float:
    checks = []
    checks.append(printmaking_ok(True, True))
    checks.append(not printmaking_ok(False, True))
    checks.append(printmaking_aux(True))
    checks.append(not printmaking_aux(False))
    checks.append(True)  # visual arts canon
    return float(sum(checks) / len(checks))


def bench_printmaking(seed: int = 0) -> dict[str, float]:
    return {"synthetic_printmaking": _bench_printmaking(seed)}
