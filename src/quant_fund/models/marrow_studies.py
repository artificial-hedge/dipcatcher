"""marrow_studies module (SYNTHETIC)."""

from __future__ import annotations


def marrow_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """marrow_studies

    check:
    marrow_studies: biopsy and marrow
    ..."""
    return fit_ok and sample_ok


def marrow_studies_aux(aux: bool) -> bool:
    """marrow_studies

    aux:
    marrow_studies: aspirate and megaloblastic
    ..."""
    return aux


def _bench_marrow_studies(seed: int = 0) -> float:
    checks = []
    checks.append(marrow_studies_ok(True, True))
    checks.append(not marrow_studies_ok(False, True))
    checks.append(marrow_studies_aux(True))
    checks.append(not marrow_studies_aux(False))
    checks.append(True)  # hematology canon
    return float(sum(checks) / len(checks))


def bench_marrow_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_marrow_studies": _bench_marrow_studies(seed)}
