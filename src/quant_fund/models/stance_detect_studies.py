"""stance_detect_studies module (SYNTHETIC)."""

from __future__ import annotations


def stance_detect_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """stance_detect_studies

    check:
    stance_detect_studies: stance-detection metrics
    """
    return fit_ok and sample_ok


def stance_detect_studies_aux(aux: bool) -> bool:
    """stance_detect_studies

    aux:
    stance_detect_studies: posts, targets, labels, and accuracies
    """
    return aux


def _bench_stance_detect_studies(seed: int = 0) -> float:
    checks = []
    checks.append(stance_detect_studies_ok(True, True))
    checks.append(not stance_detect_studies_ok(False, True))
    checks.append(stance_detect_studies_aux(True))
    checks.append(not stance_detect_studies_aux(False))
    checks.append(True)  # fake-news canon
    return float(sum(checks) / len(checks))


def bench_stance_detect_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stance_detect_studies": _bench_stance_detect_studies(seed)}
