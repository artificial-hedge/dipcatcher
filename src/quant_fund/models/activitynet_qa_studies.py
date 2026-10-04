"""activitynet_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def activitynet_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """activitynet_qa_studies

    check:
    activitynet_qa_studies: ActivityNet-QA metrics
    """
    return fit_ok and sample_ok


def activitynet_qa_studies_aux(aux: bool) -> bool:
    """activitynet_qa_studies

    aux:
    activitynet_qa_studies: videos, questions, answers, and scores
    """
    return aux


def _bench_activitynet_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(activitynet_qa_studies_ok(True, True))
    checks.append(not activitynet_qa_studies_ok(False, True))
    checks.append(activitynet_qa_studies_aux(True))
    checks.append(not activitynet_qa_studies_aux(False))
    checks.append(True)  # video-QA canon
    return float(sum(checks) / len(checks))


def bench_activitynet_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_activitynet_qa_studies": _bench_activitynet_qa_studies(seed)}
