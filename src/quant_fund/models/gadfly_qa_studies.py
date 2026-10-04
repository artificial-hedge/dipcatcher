"""gadfly_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gadfly_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gadfly_qa_studies

    check:
    gadfly_qa_studies: GadflyQA metrics
    """
    return fit_ok and sample_ok


def gadfly_qa_studies_aux(aux: bool) -> bool:
    """gadfly_qa_studies

    aux:
    gadfly_qa_studies: gadfly petrels, islets, answers, and scores
    """
    return aux


def _bench_gadfly_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gadfly_qa_studies_ok(True, True))
    checks.append(not gadfly_qa_studies_ok(False, True))
    checks.append(gadfly_qa_studies_aux(True))
    checks.append(not gadfly_qa_studies_aux(False))
    checks.append(True)  # pelagic canon
    return float(sum(checks) / len(checks))


def bench_gadfly_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gadfly_qa_studies": _bench_gadfly_qa_studies(seed)}
