"""leopard_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def leopard_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """leopard_qa_studies

    check:
    leopard_qa_studies: LeopardQA metrics
    """
    return fit_ok and sample_ok


def leopard_qa_studies_aux(aux: bool) -> bool:
    """leopard_qa_studies

    aux:
    leopard_qa_studies: leopards, rosettes, answers, and scores
    """
    return aux


def _bench_leopard_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(leopard_qa_studies_ok(True, True))
    checks.append(not leopard_qa_studies_ok(False, True))
    checks.append(leopard_qa_studies_aux(True))
    checks.append(not leopard_qa_studies_aux(False))
    checks.append(True)  # predator canon
    return float(sum(checks) / len(checks))


def bench_leopard_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_leopard_qa_studies": _bench_leopard_qa_studies(seed)}
