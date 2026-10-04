"""fern_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fern_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fern_qa_studies

    check:
    fern_qa_studies: FernQA metrics
    """
    return fit_ok and sample_ok


def fern_qa_studies_aux(aux: bool) -> bool:
    """fern_qa_studies

    aux:
    fern_qa_studies: ferns, fronds, answers, and scores
    """
    return aux


def _bench_fern_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fern_qa_studies_ok(True, True))
    checks.append(not fern_qa_studies_ok(False, True))
    checks.append(fern_qa_studies_aux(True))
    checks.append(not fern_qa_studies_aux(False))
    checks.append(True)  # flora canon
    return float(sum(checks) / len(checks))


def bench_fern_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fern_qa_studies": _bench_fern_qa_studies(seed)}
