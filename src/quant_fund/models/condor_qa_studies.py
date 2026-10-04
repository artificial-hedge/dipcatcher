"""condor_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def condor_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """condor_qa_studies

    check:
    condor_qa_studies: CondorQA metrics
    """
    return fit_ok and sample_ok


def condor_qa_studies_aux(aux: bool) -> bool:
    """condor_qa_studies

    aux:
    condor_qa_studies: condors, carrion, answers, and scores
    """
    return aux


def _bench_condor_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(condor_qa_studies_ok(True, True))
    checks.append(not condor_qa_studies_ok(False, True))
    checks.append(condor_qa_studies_aux(True))
    checks.append(not condor_qa_studies_aux(False))
    checks.append(True)  # raptor canon
    return float(sum(checks) / len(checks))


def bench_condor_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_condor_qa_studies": _bench_condor_qa_studies(seed)}
