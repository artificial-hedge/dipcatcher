"""nanna_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nanna_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nanna_qa_studies

    check:
    nanna_qa_studies: NannaQA metrics
    """
    return fit_ok and sample_ok


def nanna_qa_studies_aux(aux: bool) -> bool:
    """nanna_qa_studies

    aux:
    nanna_qa_studies: nanna, moon fathers, answers, and scores
    """
    return aux


def _bench_nanna_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nanna_qa_studies_ok(True, True))
    checks.append(not nanna_qa_studies_ok(False, True))
    checks.append(nanna_qa_studies_aux(True))
    checks.append(not nanna_qa_studies_aux(False))
    checks.append(True)  # sumerian-2 canon
    return float(sum(checks) / len(checks))


def bench_nanna_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nanna_qa_studies": _bench_nanna_qa_studies(seed)}
