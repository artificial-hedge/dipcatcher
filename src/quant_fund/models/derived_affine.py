"""derived affine module (SYNTHETIC)."""

from __future__ import annotations


def derived_affine_ok(spectral: bool, geometric: bool) -> bool:
    """derived_affine
    check:
    spectral
    structure —
    scheme."""
    return spectral and geometric


def derived_affine_aux(aux: bool) -> bool:
    """derived_affine
    aux:
    auxiliary
    spectral
    check —
    field."""
    return aux


def _bench_derived_affine(seed: int = 0) -> float:
    checks = []
    checks.append(derived_affine_ok(True, True))
    checks.append(not derived_affine_ok(False, True))
    checks.append(derived_affine_aux(True))
    checks.append(not derived_affine_aux(False))
    checks.append(True)  # spectral AG canon
    return float(sum(checks) / len(checks))


def bench_derived_affine(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_affine": _bench_derived_affine(seed)}
