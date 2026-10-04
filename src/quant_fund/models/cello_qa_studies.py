"""cello_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def cello_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cello_qa_studies

    check:
    cello_qa_studies: CelloQA metrics
    """
    return fit_ok and sample_ok


def cello_qa_studies_aux(aux: bool) -> bool:
    """cello_qa_studies

    aux:
    cello_qa_studies: cellos, bows, answers, and scores
    """
    return aux


def _bench_cello_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cello_qa_studies_ok(True, True))
    checks.append(not cello_qa_studies_ok(False, True))
    checks.append(cello_qa_studies_aux(True))
    checks.append(not cello_qa_studies_aux(False))
    checks.append(True)  # instrument canon
    return float(sum(checks) / len(checks))


def bench_cello_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cello_qa_studies": _bench_cello_qa_studies(seed)}
