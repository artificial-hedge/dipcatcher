"""sciq_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def sciq_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sciq_lite_studies

    check:
    sciq_lite_studies: SciQ metrics
    """
    return fit_ok and sample_ok


def sciq_lite_studies_aux(aux: bool) -> bool:
    """sciq_lite_studies

    aux:
    sciq_lite_studies: questions, distractors, supports, and accuracies
    """
    return aux


def _bench_sciq_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sciq_lite_studies_ok(True, True))
    checks.append(not sciq_lite_studies_ok(False, True))
    checks.append(sciq_lite_studies_aux(True))
    checks.append(not sciq_lite_studies_aux(False))
    checks.append(True)  # MC-eval canon
    return float(sum(checks) / len(checks))


def bench_sciq_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sciq_lite_studies": _bench_sciq_lite_studies(seed)}
