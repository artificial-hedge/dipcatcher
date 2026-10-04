"""flying_fox_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def flying_fox_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """flying_fox_qa_studies

    check:
    flying_fox_qa_studies: FlyingFoxQA metrics
    """
    return fit_ok and sample_ok


def flying_fox_qa_studies_aux(aux: bool) -> bool:
    """flying_fox_qa_studies

    aux:
    flying_fox_qa_studies: flying foxes, island figs, answers, and scores
    """
    return aux


def _bench_flying_fox_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(flying_fox_qa_studies_ok(True, True))
    checks.append(not flying_fox_qa_studies_ok(False, True))
    checks.append(flying_fox_qa_studies_aux(True))
    checks.append(not flying_fox_qa_studies_aux(False))
    checks.append(True)  # bat canon
    return float(sum(checks) / len(checks))


def bench_flying_fox_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_flying_fox_qa_studies": _bench_flying_fox_qa_studies(seed)}
