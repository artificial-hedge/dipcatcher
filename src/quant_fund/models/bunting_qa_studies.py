"""bunting_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bunting_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bunting_qa_studies

    check:
    bunting_qa_studies: BuntingQA metrics
    """
    return fit_ok and sample_ok


def bunting_qa_studies_aux(aux: bool) -> bool:
    """bunting_qa_studies

    aux:
    bunting_qa_studies: buntings, fields, answers, and scores
    """
    return aux


def _bench_bunting_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bunting_qa_studies_ok(True, True))
    checks.append(not bunting_qa_studies_ok(False, True))
    checks.append(bunting_qa_studies_aux(True))
    checks.append(not bunting_qa_studies_aux(False))
    checks.append(True)  # songbird-2 canon
    return float(sum(checks) / len(checks))


def bench_bunting_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bunting_qa_studies": _bench_bunting_qa_studies(seed)}
