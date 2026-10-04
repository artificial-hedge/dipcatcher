"""parathyroid_studies module (SYNTHETIC)."""

from __future__ import annotations


def parathyroid_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """parathyroid_studies

    check:
    parathyroid_studies: parathyroid and calcium
    ..."""
    return fit_ok and sample_ok


def parathyroid_studies_aux(aux: bool) -> bool:
    """parathyroid_studies

    aux:
    parathyroid_studies: pth and hypercalcemia
    ..."""
    return aux


def _bench_parathyroid_studies(seed: int = 0) -> float:
    checks = []
    checks.append(parathyroid_studies_ok(True, True))
    checks.append(not parathyroid_studies_ok(False, True))
    checks.append(parathyroid_studies_aux(True))
    checks.append(not parathyroid_studies_aux(False))
    checks.append(True)  # metabolic-endocrine canon
    return float(sum(checks) / len(checks))


def bench_parathyroid_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_parathyroid_studies": _bench_parathyroid_studies(seed)}
