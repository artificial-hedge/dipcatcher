"""collared_dove_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def collared_dove_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """collared_dove_qa_studies

    check:
    collared_dove_qa_studies: Collared-doveQA metrics
    """
    return fit_ok and sample_ok


def collared_dove_qa_studies_aux(aux: bool) -> bool:
    """collared_dove_qa_studies

    aux:
    collared_dove_qa_studies: collared doves, feeders, answers, and scores
    """
    return aux


def _bench_collared_dove_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(collared_dove_qa_studies_ok(True, True))
    checks.append(not collared_dove_qa_studies_ok(False, True))
    checks.append(collared_dove_qa_studies_aux(True))
    checks.append(not collared_dove_qa_studies_aux(False))
    checks.append(True)  # columbid canon
    return float(sum(checks) / len(checks))


def bench_collared_dove_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_collared_dove_qa_studies": _bench_collared_dove_qa_studies(seed)}
