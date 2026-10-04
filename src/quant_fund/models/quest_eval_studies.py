"""quest_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def quest_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """quest_eval_studies

    check:
    quest_eval_studies: QuestEval QA-metrics
    """
    return fit_ok and sample_ok


def quest_eval_studies_aux(aux: bool) -> bool:
    """quest_eval_studies

    aux:
    quest_eval_studies: questions, answers, sources, and scores
    """
    return aux


def _bench_quest_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(quest_eval_studies_ok(True, True))
    checks.append(not quest_eval_studies_ok(False, True))
    checks.append(quest_eval_studies_aux(True))
    checks.append(not quest_eval_studies_aux(False))
    checks.append(True)  # faithfulness-eval canon
    return float(sum(checks) / len(checks))


def bench_quest_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quest_eval_studies": _bench_quest_eval_studies(seed)}
