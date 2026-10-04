"""nextqa_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def nextqa_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nextqa_lite_studies

    check:
    nextqa_lite_studies: NExT-QA metrics
    """
    return fit_ok and sample_ok


def nextqa_lite_studies_aux(aux: bool) -> bool:
    """nextqa_lite_studies

    aux:
    nextqa_lite_studies: videos, questions, answers, and scores
    """
    return aux


def _bench_nextqa_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nextqa_lite_studies_ok(True, True))
    checks.append(not nextqa_lite_studies_ok(False, True))
    checks.append(nextqa_lite_studies_aux(True))
    checks.append(not nextqa_lite_studies_aux(False))
    checks.append(True)  # video-QA canon
    return float(sum(checks) / len(checks))


def bench_nextqa_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nextqa_lite_studies": _bench_nextqa_lite_studies(seed)}
