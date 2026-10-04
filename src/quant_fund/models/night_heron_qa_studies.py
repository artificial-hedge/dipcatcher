"""night_heron_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def night_heron_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """night_heron_qa_studies

    check:
    night_heron_qa_studies: Night-heronQA metrics
    """
    return fit_ok and sample_ok


def night_heron_qa_studies_aux(aux: bool) -> bool:
    """night_heron_qa_studies

    aux:
    night_heron_qa_studies: night herons, mangroves, answers, and scores
    """
    return aux


def _bench_night_heron_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(night_heron_qa_studies_ok(True, True))
    checks.append(not night_heron_qa_studies_ok(False, True))
    checks.append(night_heron_qa_studies_aux(True))
    checks.append(not night_heron_qa_studies_aux(False))
    checks.append(True)  # heron canon
    return float(sum(checks) / len(checks))


def bench_night_heron_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_night_heron_qa_studies": _bench_night_heron_qa_studies(seed)}
