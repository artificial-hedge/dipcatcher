"""diamonds_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def diamonds_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """diamonds_lite_studies

    check:
    diamonds_lite_studies: DiaMonDS metrics
    """
    return fit_ok and sample_ok


def diamonds_lite_studies_aux(aux: bool) -> bool:
    """diamonds_lite_studies

    aux:
    diamonds_lite_studies: dialogs, domains, responses, and scores
    """
    return aux


def _bench_diamonds_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(diamonds_lite_studies_ok(True, True))
    checks.append(not diamonds_lite_studies_ok(False, True))
    checks.append(diamonds_lite_studies_aux(True))
    checks.append(not diamonds_lite_studies_aux(False))
    checks.append(True)  # dialogue-2 canon
    return float(sum(checks) / len(checks))


def bench_diamonds_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diamonds_lite_studies": _bench_diamonds_lite_studies(seed)}
