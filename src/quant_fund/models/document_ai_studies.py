"""document_ai_studies module (SYNTHETIC)."""

from __future__ import annotations


def document_ai_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """document_ai_studies

    check:
    document_ai_studies: layout-aware parsing and OCR-free reading/bboxes and tables
    """
    return fit_ok and sample_ok


def document_ai_studies_aux(aux: bool) -> bool:
    """document_ai_studies

    aux:
    document_ai_studies: reading order and structure extraction/forms and fields
    """
    return aux


def _bench_document_ai_studies(seed: int = 0) -> float:
    checks = []
    checks.append(document_ai_studies_ok(True, True))
    checks.append(not document_ai_studies_ok(False, True))
    checks.append(document_ai_studies_aux(True))
    checks.append(not document_ai_studies_aux(False))
    checks.append(True)  # omni-modal canon
    return float(sum(checks) / len(checks))


def bench_document_ai_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_document_ai_studies": _bench_document_ai_studies(seed)}
