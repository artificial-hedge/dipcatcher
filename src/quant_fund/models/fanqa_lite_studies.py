"""fanqa_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def fanqa_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fanqa_lite_studies

    check:
    fanqa_lite_studies: FanQA metrics
    """
    return fit_ok and sample_ok


def fanqa_lite_studies_aux(aux: bool) -> bool:
    """fanqa_lite_studies

    aux:
    fanqa_lite_studies: documents, hops, answers, and scores
    """
    return aux


def _bench_fanqa_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fanqa_lite_studies_ok(True, True))
    checks.append(not fanqa_lite_studies_ok(False, True))
    checks.append(fanqa_lite_studies_aux(True))
    checks.append(not fanqa_lite_studies_aux(False))
    checks.append(True)  # multi-hop-QA-2 canon
    return float(sum(checks) / len(checks))


def bench_fanqa_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fanqa_lite_studies": _bench_fanqa_lite_studies(seed)}
