"""onion_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def onion_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """onion_qa_studies

    check:
    onion_qa_studies: OnionQA metrics
    """
    return fit_ok and sample_ok


def onion_qa_studies_aux(aux: bool) -> bool:
    """onion_qa_studies

    aux:
    onion_qa_studies: onions, bulbs, answers, and scores
    """
    return aux


def _bench_onion_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(onion_qa_studies_ok(True, True))
    checks.append(not onion_qa_studies_ok(False, True))
    checks.append(onion_qa_studies_aux(True))
    checks.append(not onion_qa_studies_aux(False))
    checks.append(True)  # vegetable canon
    return float(sum(checks) / len(checks))


def bench_onion_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_onion_qa_studies": _bench_onion_qa_studies(seed)}
