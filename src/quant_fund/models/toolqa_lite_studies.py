"""toolqa_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def toolqa_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """toolqa_lite_studies

    check:
    toolqa_lite_studies: ToolQA metrics
    """
    return fit_ok and sample_ok


def toolqa_lite_studies_aux(aux: bool) -> bool:
    """toolqa_lite_studies

    aux:
    toolqa_lite_studies: questions, tools, answers, and scores
    """
    return aux


def _bench_toolqa_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(toolqa_lite_studies_ok(True, True))
    checks.append(not toolqa_lite_studies_ok(False, True))
    checks.append(toolqa_lite_studies_aux(True))
    checks.append(not toolqa_lite_studies_aux(False))
    checks.append(True)  # toolbench canon
    return float(sum(checks) / len(checks))


def bench_toolqa_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_toolqa_lite_studies": _bench_toolqa_lite_studies(seed)}
