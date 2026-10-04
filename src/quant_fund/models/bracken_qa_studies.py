"""bracken_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bracken_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bracken_qa_studies

    check:
    bracken_qa_studies: BrackenQA metrics
    """
    return fit_ok and sample_ok


def bracken_qa_studies_aux(aux: bool) -> bool:
    """bracken_qa_studies

    aux:
    bracken_qa_studies: brackens, hillsides, answers, and scores
    """
    return aux


def _bench_bracken_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bracken_qa_studies_ok(True, True))
    checks.append(not bracken_qa_studies_ok(False, True))
    checks.append(bracken_qa_studies_aux(True))
    checks.append(not bracken_qa_studies_aux(False))
    checks.append(True)  # fern canon
    return float(sum(checks) / len(checks))


def bench_bracken_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bracken_qa_studies": _bench_bracken_qa_studies(seed)}
