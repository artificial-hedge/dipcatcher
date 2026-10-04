"""fable_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fable_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fable_qa_studies

    check:
    fable_qa_studies: FableQA metrics
    """
    return fit_ok and sample_ok


def fable_qa_studies_aux(aux: bool) -> bool:
    """fable_qa_studies

    aux:
    fable_qa_studies: fables, morals, answers, and scores
    """
    return aux


def _bench_fable_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fable_qa_studies_ok(True, True))
    checks.append(not fable_qa_studies_ok(False, True))
    checks.append(fable_qa_studies_aux(True))
    checks.append(not fable_qa_studies_aux(False))
    checks.append(True)  # narrative-genre canon
    return float(sum(checks) / len(checks))


def bench_fable_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fable_qa_studies": _bench_fable_qa_studies(seed)}
