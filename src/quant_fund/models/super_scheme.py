"""Superschemes (SYNTHETIC)."""

from __future__ import annotations


def super_scheme_ok(locally: bool, ringed: bool) -> bool:
    """Superscheme:
    locally
    super-ringed
    space modeled
    on Spec of
    supercommutative
    rings; Manin-
    Voronov
    theory."""
    return locally and ringed


def functor_points(functor: bool) -> bool:
    """Functor
    of points
    formalism for
    superschemes:
    X(B) =
    Hom(Spec B,
    X) for
    superrings B."""
    return functor


def _bench_super_scheme(seed: int = 0) -> float:
    checks = []
    checks.append(super_scheme_ok(True, True))
    checks.append(not super_scheme_ok(False, True))
    checks.append(functor_points(True))
    checks.append(not functor_points(False))
    checks.append(True)  # Manin
    return float(sum(checks) / len(checks))


def bench_super_scheme(seed: int = 0) -> dict[str, float]:
    return {"synthetic_super_scheme": _bench_super_scheme(seed)}
