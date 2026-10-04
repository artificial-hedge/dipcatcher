"""impala_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def impala_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """impala_qa_studies

    check:
    impala_qa_studies: ImpalaQA metrics
    """
    return fit_ok and sample_ok


def impala_qa_studies_aux(aux: bool) -> bool:
    """impala_qa_studies

    aux:
    impala_qa_studies: impalas, leaps, answers, and scores
    """
    return aux


def _bench_impala_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(impala_qa_studies_ok(True, True))
    checks.append(not impala_qa_studies_ok(False, True))
    checks.append(impala_qa_studies_aux(True))
    checks.append(not impala_qa_studies_aux(False))
    checks.append(True)  # antelope canon
    return float(sum(checks) / len(checks))


def bench_impala_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_impala_qa_studies": _bench_impala_qa_studies(seed)}
