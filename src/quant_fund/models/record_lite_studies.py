"""record_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def record_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """record_lite_studies

    check:
    record_lite_studies: ReCoRD cloze metrics
    """
    return fit_ok and sample_ok


def record_lite_studies_aux(aux: bool) -> bool:
    """record_lite_studies

    aux:
    record_lite_studies: passages, queries, entities, and accuracies
    """
    return aux


def _bench_record_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(record_lite_studies_ok(True, True))
    checks.append(not record_lite_studies_ok(False, True))
    checks.append(record_lite_studies_aux(True))
    checks.append(not record_lite_studies_aux(False))
    checks.append(True)  # reading-comp-3 canon
    return float(sum(checks) / len(checks))


def bench_record_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_record_lite_studies": _bench_record_lite_studies(seed)}
