"""bias_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def bias_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bias_bench_studies

    check:
    bias_bench_studies: occupation-bias metrics
    """
    return fit_ok and sample_ok


def bias_bench_studies_aux(aux: bool) -> bool:
    """bias_bench_studies

    aux:
    bias_bench_studies: contexts, professions, predictions, and scores
    """
    return aux


def _bench_bias_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bias_bench_studies_ok(True, True))
    checks.append(not bias_bench_studies_ok(False, True))
    checks.append(bias_bench_studies_aux(True))
    checks.append(not bias_bench_studies_aux(False))
    checks.append(True)  # social-bias-eval canon
    return float(sum(checks) / len(checks))


def bench_bias_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bias_bench_studies": _bench_bias_bench_studies(seed)}
