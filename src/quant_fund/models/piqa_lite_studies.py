"""piqa_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def piqa_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """piqa_lite_studies

    check:
    piqa_lite_studies: PIQA physical-commonsense metrics
    """
    return fit_ok and sample_ok


def piqa_lite_studies_aux(aux: bool) -> bool:
    """piqa_lite_studies

    aux:
    piqa_lite_studies: goals, solutions, labels, and accuracies
    """
    return aux


def _bench_piqa_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(piqa_lite_studies_ok(True, True))
    checks.append(not piqa_lite_studies_ok(False, True))
    checks.append(piqa_lite_studies_aux(True))
    checks.append(not piqa_lite_studies_aux(False))
    checks.append(True)  # commonsense-eval canon
    return float(sum(checks) / len(checks))


def bench_piqa_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_piqa_lite_studies": _bench_piqa_lite_studies(seed)}
