"""gpt_score_studies module (SYNTHETIC)."""

from __future__ import annotations


def gpt_score_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gpt_score_studies

    check:
    gpt_score_studies: G-Eval rubric metrics
    """
    return fit_ok and sample_ok


def gpt_score_studies_aux(aux: bool) -> bool:
    """gpt_score_studies

    aux:
    gpt_score_studies: tasks, responses, aspects, and scores
    """
    return aux


def _bench_gpt_score_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gpt_score_studies_ok(True, True))
    checks.append(not gpt_score_studies_ok(False, True))
    checks.append(gpt_score_studies_aux(True))
    checks.append(not gpt_score_studies_aux(False))
    checks.append(True)  # LLM-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_gpt_score_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gpt_score_studies": _bench_gpt_score_studies(seed)}
