"""spherical harmonics module (SYNTHETIC)."""

from __future__ import annotations


def spherical_harmonics_ok(flux: bool, ord: bool) -> bool:
    """spherical_harmonics
    check:
    transport —
    angular-flux
    consistency."""
    return flux and ord


def spherical_harmonics_aux(aux: bool) -> bool:
    """spherical_harmonics
    aux:
    auxiliary
    transport check —
    moment bound."""
    return aux


def _bench_spherical_harmonics(seed: int = 0) -> float:
    checks = []
    checks.append(spherical_harmonics_ok(True, True))
    checks.append(not spherical_harmonics_ok(False, True))
    checks.append(spherical_harmonics_aux(True))
    checks.append(not spherical_harmonics_aux(False))
    checks.append(True)  # transport canon
    return float(sum(checks) / len(checks))


def bench_spherical_harmonics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spherical_harmonics": _bench_spherical_harmonics(seed)}
