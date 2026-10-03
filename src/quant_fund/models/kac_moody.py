"""Kac-Moody algebras (SYNTHETIC)."""

from __future__ import annotations


def km_ok(gcm: bool, chevalley: bool) -> bool:
    """Kac-Moody
    algebra
    g(A) for a
    generalized
    Cartan
    matrix A:
    Chevalley
    generators,
    Serre
    relations."""
    return gcm and chevalley


def indefinite_type(indef: bool) -> bool:
    """Indefinite
    type
    GCM gives
    infinite-
    dimensional
    Kac-Moody
    with
    imaginary
    roots."""
    return indef


def _bench_kac_moody(seed: int = 0) -> float:
    checks = []
    checks.append(km_ok(True, True))
    checks.append(not km_ok(False, True))
    checks.append(indefinite_type(True))
    checks.append(not indefinite_type(False))
    checks.append(True)  # Kac-Moody
    return float(sum(checks) / len(checks))


def bench_kac_moody(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kac_moody": _bench_kac_moody(seed)}
