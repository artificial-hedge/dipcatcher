"""early_exit_studies module (SYNTHETIC)."""

from __future__ import annotations


def early_exit_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """early_exit_studies

    check:
    early_exit_studies: layer skipping and confidence exits/depth and speed
    """
    return fit_ok and sample_ok


def early_exit_studies_aux(aux: bool) -> bool:
    """early_exit_studies

    aux:
    early_exit_studies: calibrated exits and guaranteed quality/thresholds and accuracy
    """
    return aux


def _bench_early_exit_studies(seed: int = 0) -> float:
    checks = []
    checks.append(early_exit_studies_ok(True, True))
    checks.append(not early_exit_studies_ok(False, True))
    checks.append(early_exit_studies_aux(True))
    checks.append(not early_exit_studies_aux(False))
    checks.append(True)  # LLM-serving canon
    return float(sum(checks) / len(checks))


def bench_early_exit_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_early_exit_studies": _bench_early_exit_studies(seed)}
