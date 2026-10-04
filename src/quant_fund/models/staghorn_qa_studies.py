"""staghorn_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def staghorn_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """staghorn_qa_studies

    check:
    staghorn_qa_studies: StaghornQA metrics
    """
    return fit_ok and sample_ok


def staghorn_qa_studies_aux(aux: bool) -> bool:
    """staghorn_qa_studies

    aux:
    staghorn_qa_studies: staghorns, canopies, answers, and scores
    """
    return aux


def _bench_staghorn_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(staghorn_qa_studies_ok(True, True))
    checks.append(not staghorn_qa_studies_ok(False, True))
    checks.append(staghorn_qa_studies_aux(True))
    checks.append(not staghorn_qa_studies_aux(False))
    checks.append(True)  # fern canon
    return float(sum(checks) / len(checks))


def bench_staghorn_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_staghorn_qa_studies": _bench_staghorn_qa_studies(seed)}
