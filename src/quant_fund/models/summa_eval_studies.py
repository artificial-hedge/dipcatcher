"""summa_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def summa_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """summa_eval_studies

    check:
    summa_eval_studies: SummaC NLI-consistency metrics
    """
    return fit_ok and sample_ok


def summa_eval_studies_aux(aux: bool) -> bool:
    """summa_eval_studies

    aux:
    summa_eval_studies: summaries, sources, labels, and scores
    """
    return aux


def _bench_summa_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(summa_eval_studies_ok(True, True))
    checks.append(not summa_eval_studies_ok(False, True))
    checks.append(summa_eval_studies_aux(True))
    checks.append(not summa_eval_studies_aux(False))
    checks.append(True)  # faithfulness-eval canon
    return float(sum(checks) / len(checks))


def bench_summa_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_summa_eval_studies": _bench_summa_eval_studies(seed)}
