"""rangi_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def rangi_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rangi_qa_studies

    check:
    rangi_qa_studies: RangiQA metrics
    """
    return fit_ok and sample_ok


def rangi_qa_studies_aux(aux: bool) -> bool:
    """rangi_qa_studies

    aux:
    rangi_qa_studies: rangi, sky fathers, answers, and scores
    """
    return aux


def _bench_rangi_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rangi_qa_studies_ok(True, True))
    checks.append(not rangi_qa_studies_ok(False, True))
    checks.append(rangi_qa_studies_aux(True))
    checks.append(not rangi_qa_studies_aux(False))
    checks.append(True)  # polynesian-myth canon
    return float(sum(checks) / len(checks))


def bench_rangi_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rangi_qa_studies": _bench_rangi_qa_studies(seed)}
