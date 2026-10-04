"""branwen_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def branwen_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """branwen_qa_studies

    check:
    branwen_qa_studies: BranwenQA metrics
    """
    return fit_ok and sample_ok


def branwen_qa_studies_aux(aux: bool) -> bool:
    """branwen_qa_studies

    aux:
    branwen_qa_studies: branwen, sorrow brides, answers, and scores
    """
    return aux


def _bench_branwen_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(branwen_qa_studies_ok(True, True))
    checks.append(not branwen_qa_studies_ok(False, True))
    checks.append(branwen_qa_studies_aux(True))
    checks.append(not branwen_qa_studies_aux(False))
    checks.append(True)  # welsh-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_branwen_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_branwen_qa_studies": _bench_branwen_qa_studies(seed)}
