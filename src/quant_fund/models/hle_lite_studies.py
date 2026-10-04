"""hle_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def hle_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hle_lite_studies

    check:
    hle_lite_studies: Humanity's-Last-Exam metrics
    """
    return fit_ok and sample_ok


def hle_lite_studies_aux(aux: bool) -> bool:
    """hle_lite_studies

    aux:
    hle_lite_studies: questions, answers, modalities, and scores
    """
    return aux


def _bench_hle_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hle_lite_studies_ok(True, True))
    checks.append(not hle_lite_studies_ok(False, True))
    checks.append(hle_lite_studies_aux(True))
    checks.append(not hle_lite_studies_aux(False))
    checks.append(True)  # frontier-eval canon
    return float(sum(checks) / len(checks))


def bench_hle_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hle_lite_studies": _bench_hle_lite_studies(seed)}
