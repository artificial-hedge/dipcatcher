"""field_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def field_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """field_qa_studies

    check:
    field_qa_studies: FieldQA metrics
    """
    return fit_ok and sample_ok


def field_qa_studies_aux(aux: bool) -> bool:
    """field_qa_studies

    aux:
    field_qa_studies: fields, crops, answers, and scores
    """
    return aux


def _bench_field_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(field_qa_studies_ok(True, True))
    checks.append(not field_qa_studies_ok(False, True))
    checks.append(field_qa_studies_aux(True))
    checks.append(not field_qa_studies_aux(False))
    checks.append(True)  # terrain-2 canon
    return float(sum(checks) / len(checks))


def bench_field_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_field_qa_studies": _bench_field_qa_studies(seed)}
