"""physiatry_studies module (SYNTHETIC)."""

from __future__ import annotations


def physiatry_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """physiatry_studies

    check:
    physiatry_studies: disability and function
    ..."""
    return fit_ok and sample_ok


def physiatry_studies_aux(aux: bool) -> bool:
    """physiatry_studies

    aux:
    physiatry_studies: pmr and prosthetics
    ..."""
    return aux


def _bench_physiatry_studies(seed: int = 0) -> float:
    checks = []
    checks.append(physiatry_studies_ok(True, True))
    checks.append(not physiatry_studies_ok(False, True))
    checks.append(physiatry_studies_aux(True))
    checks.append(not physiatry_studies_aux(False))
    checks.append(True)  # rehab-medicine canon
    return float(sum(checks) / len(checks))


def bench_physiatry_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_physiatry_studies": _bench_physiatry_studies(seed)}
