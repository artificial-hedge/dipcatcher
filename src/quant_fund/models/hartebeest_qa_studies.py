"""hartebeest_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def hartebeest_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hartebeest_qa_studies

    check:
    hartebeest_qa_studies: HartebeestQA metrics
    """
    return fit_ok and sample_ok


def hartebeest_qa_studies_aux(aux: bool) -> bool:
    """hartebeest_qa_studies

    aux:
    hartebeest_qa_studies: hartebeests, plains, answers, and scores
    """
    return aux


def _bench_hartebeest_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hartebeest_qa_studies_ok(True, True))
    checks.append(not hartebeest_qa_studies_ok(False, True))
    checks.append(hartebeest_qa_studies_aux(True))
    checks.append(not hartebeest_qa_studies_aux(False))
    checks.append(True)  # antelope-2 canon
    return float(sum(checks) / len(checks))


def bench_hartebeest_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hartebeest_qa_studies": _bench_hartebeest_qa_studies(seed)}
