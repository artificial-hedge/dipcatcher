"""race_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def race_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """race_lite_studies

    check:
    race_lite_studies: RACE reading metrics
    """
    return fit_ok and sample_ok


def race_lite_studies_aux(aux: bool) -> bool:
    """race_lite_studies

    aux:
    race_lite_studies: passages, questions, options, and accuracies
    """
    return aux


def _bench_race_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(race_lite_studies_ok(True, True))
    checks.append(not race_lite_studies_ok(False, True))
    checks.append(race_lite_studies_aux(True))
    checks.append(not race_lite_studies_aux(False))
    checks.append(True)  # MC-eval canon
    return float(sum(checks) / len(checks))


def bench_race_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_race_lite_studies": _bench_race_lite_studies(seed)}
