"""alfworld_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def alfworld_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """alfworld_lite_studies

    check:
    alfworld_lite_studies: ALFWorld metrics
    """
    return fit_ok and sample_ok


def alfworld_lite_studies_aux(aux: bool) -> bool:
    """alfworld_lite_studies

    aux:
    alfworld_lite_studies: tasks, actions, feedback, and scores
    """
    return aux


def _bench_alfworld_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(alfworld_lite_studies_ok(True, True))
    checks.append(not alfworld_lite_studies_ok(False, True))
    checks.append(alfworld_lite_studies_aux(True))
    checks.append(not alfworld_lite_studies_aux(False))
    checks.append(True)  # embodied-game canon
    return float(sum(checks) / len(checks))


def bench_alfworld_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_alfworld_lite_studies": _bench_alfworld_lite_studies(seed)}
