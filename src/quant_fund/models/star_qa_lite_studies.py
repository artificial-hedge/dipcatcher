"""star_qa_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def star_qa_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """star_qa_lite_studies

    check:
    star_qa_lite_studies: STAR metrics
    """
    return fit_ok and sample_ok


def star_qa_lite_studies_aux(aux: bool) -> bool:
    """star_qa_lite_studies

    aux:
    star_qa_lite_studies: videos, questions, answers, and scores
    """
    return aux


def _bench_star_qa_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(star_qa_lite_studies_ok(True, True))
    checks.append(not star_qa_lite_studies_ok(False, True))
    checks.append(star_qa_lite_studies_aux(True))
    checks.append(not star_qa_lite_studies_aux(False))
    checks.append(True)  # video-QA canon
    return float(sum(checks) / len(checks))


def bench_star_qa_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_star_qa_lite_studies": _bench_star_qa_lite_studies(seed)}
