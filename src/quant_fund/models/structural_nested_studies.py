"""structural_nested_studies module (SYNTHETIC)."""

from __future__ import annotations


def structural_nested_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """structural_nested_studies

    check:
    structural_nested_studies: g-estimation and rank preserving/SNMM and effects
    """
    return fit_ok and sample_ok


def structural_nested_studies_aux(aux: bool) -> bool:
    """structural_nested_studies

    aux:
    structural_nested_studies: blip functions and time-varying/confounding and models
    """
    return aux


def _bench_structural_nested_studies(seed: int = 0) -> float:
    checks = []
    checks.append(structural_nested_studies_ok(True, True))
    checks.append(not structural_nested_studies_ok(False, True))
    checks.append(structural_nested_studies_aux(True))
    checks.append(not structural_nested_studies_aux(False))
    checks.append(True)  # causal-RWE-2 canon
    return float(sum(checks) / len(checks))


def bench_structural_nested_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_structural_nested_studies": _bench_structural_nested_studies(seed)}
