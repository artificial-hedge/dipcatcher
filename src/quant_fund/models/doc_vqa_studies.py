"""doc_vqa_studies module (SYNTHETIC)."""

from __future__ import annotations


def doc_vqa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """doc_vqa_studies

    check:
    doc_vqa_studies: layout-aware document QA and OCR-free reading/pages and fields
    """
    return fit_ok and sample_ok


def doc_vqa_studies_aux(aux: bool) -> bool:
    """doc_vqa_studies

    aux:
    doc_vqa_studies: DocVQA/Donut-style form understanding/regions and keys
    """
    return aux


def _bench_doc_vqa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(doc_vqa_studies_ok(True, True))
    checks.append(not doc_vqa_studies_ok(False, True))
    checks.append(doc_vqa_studies_aux(True))
    checks.append(not doc_vqa_studies_aux(False))
    checks.append(True)  # multimodal-2 canon
    return float(sum(checks) / len(checks))


def bench_doc_vqa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_doc_vqa_studies": _bench_doc_vqa_studies(seed)}
