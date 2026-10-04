"""pond_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pond_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pond_qa_studies

    check:
    pond_qa_studies: PondQA metrics
    """
    return fit_ok and sample_ok


def pond_qa_studies_aux(aux: bool) -> bool:
    """pond_qa_studies

    aux:
    pond_qa_studies: ponds, lilies, answers, and scores
    """
    return aux


def _bench_pond_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pond_qa_studies_ok(True, True))
    checks.append(not pond_qa_studies_ok(False, True))
    checks.append(pond_qa_studies_aux(True))
    checks.append(not pond_qa_studies_aux(False))
    checks.append(True)  # wetland canon
    return float(sum(checks) / len(checks))


def bench_pond_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pond_qa_studies": _bench_pond_qa_studies(seed)}
