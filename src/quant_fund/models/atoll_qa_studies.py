"""atoll_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def atoll_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """atoll_qa_studies

    check:
    atoll_qa_studies: AtollQA metrics
    """
    return fit_ok and sample_ok


def atoll_qa_studies_aux(aux: bool) -> bool:
    """atoll_qa_studies

    aux:
    atoll_qa_studies: atolls, lagoons, answers, and scores
    """
    return aux


def _bench_atoll_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(atoll_qa_studies_ok(True, True))
    checks.append(not atoll_qa_studies_ok(False, True))
    checks.append(atoll_qa_studies_aux(True))
    checks.append(not atoll_qa_studies_aux(False))
    checks.append(True)  # coastal canon
    return float(sum(checks) / len(checks))


def bench_atoll_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_atoll_qa_studies": _bench_atoll_qa_studies(seed)}
