"""coriander_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def coriander_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """coriander_qa_studies

    check:
    coriander_qa_studies: CorianderQA metrics
    """
    return fit_ok and sample_ok


def coriander_qa_studies_aux(aux: bool) -> bool:
    """coriander_qa_studies

    aux:
    coriander_qa_studies: corianders, seeds, answers, and scores
    """
    return aux


def _bench_coriander_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(coriander_qa_studies_ok(True, True))
    checks.append(not coriander_qa_studies_ok(False, True))
    checks.append(coriander_qa_studies_aux(True))
    checks.append(not coriander_qa_studies_aux(False))
    checks.append(True)  # herb canon
    return float(sum(checks) / len(checks))


def bench_coriander_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_coriander_qa_studies": _bench_coriander_qa_studies(seed)}
