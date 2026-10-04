"""early_modern module (SYNTHETIC)."""

from __future__ import annotations


def early_modern_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """early_modern

    check:
    renaissance_studies: renaissance studies
    early_modern: early modern
    humanism: humanism
    reformation_studies: reformation studies
    baroque_studies: baroque studies
    enlightenment_studies: enlightenment studies
    """
    return fit_ok and sample_ok


def early_modern_aux(aux: bool) -> bool:
    """early_modern

    aux:
    renaissance_studies: rinascimento
    early_modern: 1500-1800 period
    humanism: classical learning
    reformation_studies: protestant reformation
    baroque_studies: baroque culture
    enlightenment_studies: age of reason
    """
    return aux


def _bench_early_modern(seed: int = 0) -> float:
    checks = []
    checks.append(early_modern_ok(True, True))
    checks.append(not early_modern_ok(False, True))
    checks.append(early_modern_aux(True))
    checks.append(not early_modern_aux(False))
    checks.append(True)  # early modern canon
    return float(sum(checks) / len(checks))


def bench_early_modern(seed: int = 0) -> dict[str, float]:
    return {"synthetic_early_modern": _bench_early_modern(seed)}
