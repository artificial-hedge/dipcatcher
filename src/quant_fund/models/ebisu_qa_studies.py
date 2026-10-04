"""ebisu_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ebisu_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ebisu_qa_studies

    check:
    ebisu_qa_studies: EbisuQA metrics
    """
    return fit_ok and sample_ok


def ebisu_qa_studies_aux(aux: bool) -> bool:
    """ebisu_qa_studies

    aux:
    ebisu_qa_studies: ebisu, fisher lords, answers, and scores
    """
    return aux


def _bench_ebisu_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ebisu_qa_studies_ok(True, True))
    checks.append(not ebisu_qa_studies_ok(False, True))
    checks.append(ebisu_qa_studies_aux(True))
    checks.append(not ebisu_qa_studies_aux(False))
    checks.append(True)  # japanese-myth-4 canon
    return float(sum(checks) / len(checks))


def bench_ebisu_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ebisu_qa_studies": _bench_ebisu_qa_studies(seed)}
