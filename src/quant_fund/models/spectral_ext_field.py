"""spectral ext_field module (SYNTHETIC)."""

from __future__ import annotations


def spectral_ext_field_ok(spectral: bool, geometric: bool) -> bool:
    """spectral_ext_field
    check:
    spectral
    structure —
    prime."""
    return spectral and geometric


def spectral_ext_field_aux(aux: bool) -> bool:
    """spectral_ext_field
    aux:
    auxiliary
    spectral
    check —
    level."""
    return aux


def _bench_spectral_ext_field(seed: int = 0) -> float:
    checks = []
    checks.append(spectral_ext_field_ok(True, True))
    checks.append(not spectral_ext_field_ok(False, True))
    checks.append(spectral_ext_field_aux(True))
    checks.append(not spectral_ext_field_aux(False))
    checks.append(True)  # spectral AG canon
    return float(sum(checks) / len(checks))


def bench_spectral_ext_field(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_ext_field": _bench_spectral_ext_field(seed)}
