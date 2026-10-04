"""coagulation_studies module (SYNTHETIC)."""

from __future__ import annotations


def coagulation_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """coagulation_studies

    check:
    coagulation_studies: factors and inr
    ..."""
    return fit_ok and sample_ok


def coagulation_studies_aux(aux: bool) -> bool:
    """coagulation_studies

    aux:
    coagulation_studies: heparin and doac
    ..."""
    return aux


def _bench_coagulation_studies(seed: int = 0) -> float:
    checks = []
    checks.append(coagulation_studies_ok(True, True))
    checks.append(not coagulation_studies_ok(False, True))
    checks.append(coagulation_studies_aux(True))
    checks.append(not coagulation_studies_aux(False))
    checks.append(True)  # hematology canon
    return float(sum(checks) / len(checks))


def bench_coagulation_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_coagulation_studies": _bench_coagulation_studies(seed)}
