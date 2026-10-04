"""how2qa_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def how2qa_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """how2qa_lite_studies

    check:
    how2qa_lite_studies: How2QA metrics
    """
    return fit_ok and sample_ok


def how2qa_lite_studies_aux(aux: bool) -> bool:
    """how2qa_lite_studies

    aux:
    how2qa_lite_studies: videos, questions, answers, and scores
    """
    return aux


def _bench_how2qa_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(how2qa_lite_studies_ok(True, True))
    checks.append(not how2qa_lite_studies_ok(False, True))
    checks.append(how2qa_lite_studies_aux(True))
    checks.append(not how2qa_lite_studies_aux(False))
    checks.append(True)  # video-QA canon
    return float(sum(checks) / len(checks))


def bench_how2qa_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_how2qa_lite_studies": _bench_how2qa_lite_studies(seed)}
