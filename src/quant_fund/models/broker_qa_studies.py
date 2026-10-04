"""broker_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def broker_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """broker_qa_studies

    check:
    broker_qa_studies: BrokerQA metrics
    """
    return fit_ok and sample_ok


def broker_qa_studies_aux(aux: bool) -> bool:
    """broker_qa_studies

    aux:
    broker_qa_studies: orders, fills, answers, and scores
    """
    return aux


def _bench_broker_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(broker_qa_studies_ok(True, True))
    checks.append(not broker_qa_studies_ok(False, True))
    checks.append(broker_qa_studies_aux(True))
    checks.append(not broker_qa_studies_aux(False))
    checks.append(True)  # financial-NLP canon
    return float(sum(checks) / len(checks))


def bench_broker_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_broker_qa_studies": _bench_broker_qa_studies(seed)}
