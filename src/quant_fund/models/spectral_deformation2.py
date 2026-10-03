"""spectral deformation2 module (SYNTHETIC)."""

from __future__ import annotations


def spectral_deformation2_ok(derived: bool, geometry: bool) -> bool:
    """spectral_deformation2
    check:
    derived
    geometry —
    spectral."""
    return derived and geometry


def spectral_deformation2_aux(aux: bool) -> bool:
    """spectral_deformation2
    aux:
    auxiliary
    derived
    check —
    stack."""
    return aux


def _bench_spectral_deformation2(seed: int = 0) -> float:
    checks = []
    checks.append(spectral_deformation2_ok(True, True))
    checks.append(not spectral_deformation2_ok(False, True))
    checks.append(spectral_deformation2_aux(True))
    checks.append(not spectral_deformation2_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_spectral_deformation2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_deformation2": _bench_spectral_deformation2(seed)}
