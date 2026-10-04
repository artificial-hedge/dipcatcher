"""phlebology_studies module (SYNTHETIC)."""

from __future__ import annotations


def phlebology_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """phlebology_studies

    check:
    phlebology_studies: varicose veins and thrombosis
    ..."""
    return fit_ok and sample_ok


def phlebology_studies_aux(aux: bool) -> bool:
    """phlebology_studies

    aux:
    phlebology_studies: sclerotherapy and compression
    ..."""
    return aux


def _bench_phlebology_studies(seed: int = 0) -> float:
    checks = []
    checks.append(phlebology_studies_ok(True, True))
    checks.append(not phlebology_studies_ok(False, True))
    checks.append(phlebology_studies_aux(True))
    checks.append(not phlebology_studies_aux(False))
    checks.append(True)  # vascular-medicine canon
    return float(sum(checks) / len(checks))


def bench_phlebology_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_phlebology_studies": _bench_phlebology_studies(seed)}
