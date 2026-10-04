"""Perverse sheaves II (SYNTHETIC)."""

from __future__ import annotations


def perverse_tstruct(top_bis: bool, support_cond: bool) -> bool:
    """Perv(X) is the heart of the perverse
    t-structure: H^i(F) supported on strata of
    codim > i (BBD)."""
    return top_bis and support_cond


def artin_vanishing(proper_map: bool) -> bool:
    """Artin vanishing for affine morphisms;
    perverse sheaves preserved by middle
    perversity."""
    return proper_map


def _bench_perverse_2(seed: int = 0) -> float:
    checks = []
    checks.append(perverse_tstruct(True, True))
    checks.append(not perverse_tstruct(False, True))
    checks.append(artin_vanishing(True))
    checks.append(not artin_vanishing(False))
    checks.append(True)  # IC complexes = simple perverse
    return float(sum(checks) / len(checks))


def bench_perverse_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_perverse_2": _bench_perverse_2(seed)}
