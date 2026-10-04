"""painting_techniques module (SYNTHETIC)."""

from __future__ import annotations


def painting_techniques_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """painting_techniques

    check:
    painting_techniques: painting techniques
    sculpture_methods: sculpture methods
    printmaking: printmaking
    art_conservation: art conservation
    art_history: art history
    visual_culture: visual culture
    """
    return fit_ok and sample_ok


def painting_techniques_aux(aux: bool) -> bool:
    """painting_techniques

    aux:
    painting_techniques: pigment and medium
    sculpture_methods: three-dimensional form
    printmaking: impression processes
    art_conservation: preservation methods
    art_history: art historical periods
    visual_culture: visual studies
    """
    return aux


def _bench_painting_techniques(seed: int = 0) -> float:
    checks = []
    checks.append(painting_techniques_ok(True, True))
    checks.append(not painting_techniques_ok(False, True))
    checks.append(painting_techniques_aux(True))
    checks.append(not painting_techniques_aux(False))
    checks.append(True)  # visual arts canon
    return float(sum(checks) / len(checks))


def bench_painting_techniques(seed: int = 0) -> dict[str, float]:
    return {"synthetic_painting_techniques": _bench_painting_techniques(seed)}
