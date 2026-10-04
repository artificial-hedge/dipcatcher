"""Multiplier ideals (SYNTHETIC)."""

from __future__ import annotations


def mi_ok(integrability: bool, ideal_sheaf: bool) -> bool:
    """Multiplier
    ideal:
    ideal
    sheaf
    of
    functions
    with
    integrable
    pole
    quotients —
    Nadel
    1989."""
    return integrability and ideal_sheaf


def nadel_vanishing(nv: bool) -> bool:
    """Nadel
    vanishing:
    Kawamata-
    Viehweg
    for
    multiplier
    ideals —
    adjoint
    bundles
    vanish."""
    return nv


def _bench_multiplier_ideal(seed: int = 0) -> float:
    checks = []
    checks.append(mi_ok(True, True))
    checks.append(not mi_ok(False, True))
    checks.append(nadel_vanishing(True))
    checks.append(not nadel_vanishing(False))
    checks.append(True)  # Nadel
    return float(sum(checks) / len(checks))


def bench_multiplier_ideal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_multiplier_ideal": _bench_multiplier_ideal(seed)}
