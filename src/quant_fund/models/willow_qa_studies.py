"""willow_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def willow_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """willow_qa_studies

    check:
    willow_qa_studies: WillowQA metrics
    """
    return fit_ok and sample_ok


def willow_qa_studies_aux(aux: bool) -> bool:
    """willow_qa_studies

    aux:
    willow_qa_studies: willows, catkins, answers, and scores
    """
    return aux


def _bench_willow_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(willow_qa_studies_ok(True, True))
    checks.append(not willow_qa_studies_ok(False, True))
    checks.append(willow_qa_studies_aux(True))
    checks.append(not willow_qa_studies_aux(False))
    checks.append(True)  # arboreal canon
    return float(sum(checks) / len(checks))


def bench_willow_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_willow_qa_studies": _bench_willow_qa_studies(seed)}
