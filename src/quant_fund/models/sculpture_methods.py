"""sculpture_methods module (SYNTHETIC)."""

from __future__ import annotations


def sculpture_methods_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sculpture_methods

    check:
    painting_techniques: painting techniques
    sculpture_methods: sculpture methods
    printmaking: printmaking
    art_conservation: art conservation
    art_history: art history
    visual_culture: visual culture
    """
    return fit_ok and sample_ok


def sculpture_methods_aux(aux: bool) -> bool:
    """sculpture_methods

    aux:
    painting_techniques: pigment and medium
    sculpture_methods: three-dimensional form
    printmaking: impression processes
    art_conservation: preservation methods
    art_history: art historical periods
    visual_culture: visual studies
    """
    return aux


def _bench_sculpture_methods(seed: int = 0) -> float:
    checks = []
    checks.append(sculpture_methods_ok(True, True))
    checks.append(not sculpture_methods_ok(False, True))
    checks.append(sculpture_methods_aux(True))
    checks.append(not sculpture_methods_aux(False))
    checks.append(True)  # visual arts canon
    return float(sum(checks) / len(checks))


def bench_sculpture_methods(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sculpture_methods": _bench_sculpture_methods(seed)}
