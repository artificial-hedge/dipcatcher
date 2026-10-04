"""Derived tangent (SYNTHETIC)."""

from __future__ import annotations


def dt_ok(derived: bool, tangent: bool) -> bool:
    """Derived
    tangent:
    tangent
    complex
    of
    derived
    space —
    Illusie
    tangent."""
    return derived and tangent


def illusie_tangent(it: bool) -> bool:
    """Illusie
    tangent:
    Illusie
    tangent
    complex —
    deformation
    tangent."""
    return it


def _bench_derived_tangent(seed: int = 0) -> float:
    checks = []
    checks.append(dt_ok(True, True))
    checks.append(not dt_ok(False, True))
    checks.append(illusie_tangent(True))
    checks.append(not illusie_tangent(False))
    checks.append(True)  # Illusie
    return float(sum(checks) / len(checks))


def bench_derived_tangent(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_tangent": _bench_derived_tangent(seed)}
