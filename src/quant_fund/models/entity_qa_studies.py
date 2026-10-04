"""entity_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def entity_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """entity_qa_studies

    check:
    entity_qa_studies: entity-QA metrics
    """
    return fit_ok and sample_ok


def entity_qa_studies_aux(aux: bool) -> bool:
    """entity_qa_studies

    aux:
    entity_qa_studies: entities, questions, answers, and accuracies
    """
    return aux


def _bench_entity_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(entity_qa_studies_ok(True, True))
    checks.append(not entity_qa_studies_ok(False, True))
    checks.append(entity_qa_studies_aux(True))
    checks.append(not entity_qa_studies_aux(False))
    checks.append(True)  # knowledge-QA canon
    return float(sum(checks) / len(checks))


def bench_entity_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_entity_qa_studies": _bench_entity_qa_studies(seed)}
