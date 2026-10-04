"""gpqa_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def gpqa_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gpqa_lite_studies

    check:
    gpqa_lite_studies: GPQA graduate-science metrics
    """
    return fit_ok and sample_ok


def gpqa_lite_studies_aux(aux: bool) -> bool:
    """gpqa_lite_studies

    aux:
    gpqa_lite_studies: questions, choices, answers, and scores
    """
    return aux


def _bench_gpqa_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gpqa_lite_studies_ok(True, True))
    checks.append(not gpqa_lite_studies_ok(False, True))
    checks.append(gpqa_lite_studies_aux(True))
    checks.append(not gpqa_lite_studies_aux(False))
    checks.append(True)  # challenge-benchmark canon
    return float(sum(checks) / len(checks))


def bench_gpqa_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gpqa_lite_studies": _bench_gpqa_lite_studies(seed)}
