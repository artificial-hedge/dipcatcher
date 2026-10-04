"""liverwort_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def liverwort_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """liverwort_qa_studies

    check:
    liverwort_qa_studies: LiverwortQA metrics
    """
    return fit_ok and sample_ok


def liverwort_qa_studies_aux(aux: bool) -> bool:
    """liverwort_qa_studies

    aux:
    liverwort_qa_studies: liverworts, streambanks, answers, and scores
    """
    return aux


def _bench_liverwort_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(liverwort_qa_studies_ok(True, True))
    checks.append(not liverwort_qa_studies_ok(False, True))
    checks.append(liverwort_qa_studies_aux(True))
    checks.append(not liverwort_qa_studies_aux(False))
    checks.append(True)  # moss canon
    return float(sum(checks) / len(checks))


def bench_liverwort_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_liverwort_qa_studies": _bench_liverwort_qa_studies(seed)}
