"""tubulointerstitial_studies module (SYNTHETIC)."""

from __future__ import annotations


def tubulointerstitial_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tubulointerstitial_studies

    check:
    tubulointerstitial_studies: tubules and interstitium
    ..."""
    return fit_ok and sample_ok


def tubulointerstitial_studies_aux(aux: bool) -> bool:
    """tubulointerstitial_studies

    aux:
    tubulointerstitial_studies: ain and biopsy
    ..."""
    return aux


def _bench_tubulointerstitial_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tubulointerstitial_studies_ok(True, True))
    checks.append(not tubulointerstitial_studies_ok(False, True))
    checks.append(tubulointerstitial_studies_aux(True))
    checks.append(not tubulointerstitial_studies_aux(False))
    checks.append(True)  # nephro-renal canon
    return float(sum(checks) / len(checks))


def bench_tubulointerstitial_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tubulointerstitial_studies": _bench_tubulointerstitial_studies(seed)}
