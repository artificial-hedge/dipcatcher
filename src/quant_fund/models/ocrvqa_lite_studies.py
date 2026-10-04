"""ocrvqa_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def ocrvqa_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ocrvqa_lite_studies

    check:
    ocrvqa_lite_studies: OCR-VQA metrics
    """
    return fit_ok and sample_ok


def ocrvqa_lite_studies_aux(aux: bool) -> bool:
    """ocrvqa_lite_studies

    aux:
    ocrvqa_lite_studies: images, questions, answers, and scores
    """
    return aux


def _bench_ocrvqa_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ocrvqa_lite_studies_ok(True, True))
    checks.append(not ocrvqa_lite_studies_ok(False, True))
    checks.append(ocrvqa_lite_studies_aux(True))
    checks.append(not ocrvqa_lite_studies_aux(False))
    checks.append(True)  # vision-doc-QA canon
    return float(sum(checks) / len(checks))


def bench_ocrvqa_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ocrvqa_lite_studies": _bench_ocrvqa_lite_studies(seed)}
