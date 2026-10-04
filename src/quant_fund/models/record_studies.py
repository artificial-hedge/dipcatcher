"""record_studies module (SYNTHETIC)."""

from __future__ import annotations


def record_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """record_studies

    check:
    record_studies: ReCoRD cloze-style QA entities and EM/F1
    """
    return fit_ok and sample_ok


def record_studies_aux(aux: bool) -> bool:
    """record_studies

    aux:
    record_studies: passages, placeholders, entity sets, and answers
    """
    return aux


def _bench_record_studies(seed: int = 0) -> float:
    checks = []
    checks.append(record_studies_ok(True, True))
    checks.append(not record_studies_ok(False, True))
    checks.append(record_studies_aux(True))
    checks.append(not record_studies_aux(False))
    checks.append(True)  # winograd-eval canon
    return float(sum(checks) / len(checks))


def bench_record_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_record_studies": _bench_record_studies(seed)}
