"""tuna_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tuna_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tuna_qa_studies

    check:
    tuna_qa_studies: TunaQA metrics
    """
    return fit_ok and sample_ok


def tuna_qa_studies_aux(aux: bool) -> bool:
    """tuna_qa_studies

    aux:
    tuna_qa_studies: tunas, schools, answers, and scores
    """
    return aux


def _bench_tuna_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tuna_qa_studies_ok(True, True))
    checks.append(not tuna_qa_studies_ok(False, True))
    checks.append(tuna_qa_studies_aux(True))
    checks.append(not tuna_qa_studies_aux(False))
    checks.append(True)  # fish canon
    return float(sum(checks) / len(checks))


def bench_tuna_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tuna_qa_studies": _bench_tuna_qa_studies(seed)}
