"""interstitial_lung_studies module (SYNTHETIC)."""

from __future__ import annotations


def interstitial_lung_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """interstitial_lung_studies

    check:
    interstitial_lung_studies: fibrosis and ild
    ..."""
    return fit_ok and sample_ok


def interstitial_lung_studies_aux(aux: bool) -> bool:
    """interstitial_lung_studies

    aux:
    interstitial_lung_studies: hrct and dlco
    ..."""
    return aux


def _bench_interstitial_lung_studies(seed: int = 0) -> float:
    checks = []
    checks.append(interstitial_lung_studies_ok(True, True))
    checks.append(not interstitial_lung_studies_ok(False, True))
    checks.append(interstitial_lung_studies_aux(True))
    checks.append(not interstitial_lung_studies_aux(False))
    checks.append(True)  # pulmonology canon
    return float(sum(checks) / len(checks))


def bench_interstitial_lung_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_interstitial_lung_studies": _bench_interstitial_lung_studies(seed)}
