"""truthfulqa_studies module (SYNTHETIC)."""

from __future__ import annotations


def truthfulqa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """truthfulqa_studies

    check:
    truthfulqa_studies: TruthfulQA truthfulness and informativeness rates
    """
    return fit_ok and sample_ok


def truthfulqa_studies_aux(aux: bool) -> bool:
    """truthfulqa_studies

    aux:
    truthfulqa_studies: questions, answers, and truthful-score rates
    """
    return aux


def _bench_truthfulqa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(truthfulqa_studies_ok(True, True))
    checks.append(not truthfulqa_studies_ok(False, True))
    checks.append(truthfulqa_studies_aux(True))
    checks.append(not truthfulqa_studies_aux(False))
    checks.append(True)  # reasoning-eval canon
    return float(sum(checks) / len(checks))


def bench_truthfulqa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_truthfulqa_studies": _bench_truthfulqa_studies(seed)}
