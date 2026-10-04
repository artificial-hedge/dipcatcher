"""horsetail_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def horsetail_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """horsetail_qa_studies

    check:
    horsetail_qa_studies: HorsetailQA metrics
    """
    return fit_ok and sample_ok


def horsetail_qa_studies_aux(aux: bool) -> bool:
    """horsetail_qa_studies

    aux:
    horsetail_qa_studies: horsetails, wetlands, answers, and scores
    """
    return aux


def _bench_horsetail_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(horsetail_qa_studies_ok(True, True))
    checks.append(not horsetail_qa_studies_ok(False, True))
    checks.append(horsetail_qa_studies_aux(True))
    checks.append(not horsetail_qa_studies_aux(False))
    checks.append(True)  # fern canon
    return float(sum(checks) / len(checks))


def bench_horsetail_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_horsetail_qa_studies": _bench_horsetail_qa_studies(seed)}
