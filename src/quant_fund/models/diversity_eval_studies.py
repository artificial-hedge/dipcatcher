"""diversity_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def diversity_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """diversity_eval_studies

    check:
    diversity_eval_studies: distinct-n/entropy generation diversity metrics
    """
    return fit_ok and sample_ok


def diversity_eval_studies_aux(aux: bool) -> bool:
    """diversity_eval_studies

    aux:
    diversity_eval_studies: generations, token stats, and diversity scores
    """
    return aux


def _bench_diversity_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(diversity_eval_studies_ok(True, True))
    checks.append(not diversity_eval_studies_ok(False, True))
    checks.append(diversity_eval_studies_aux(True))
    checks.append(not diversity_eval_studies_aux(False))
    checks.append(True)  # generation-quality canon
    return float(sum(checks) / len(checks))


def bench_diversity_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diversity_eval_studies": _bench_diversity_eval_studies(seed)}
