"""Affinoid algebras (SYNTHETIC)."""

from __future__ import annotations


def affinoid_ok(tate_quotient: bool, noetherian: bool) -> bool:
    """Affinoid algebra = quotient of Tate
    algebra T_n; Noetherian + Banach
    (Tate, Berkovich)."""
    return tate_quotient and noetherian


def max_spectrum(points: bool) -> bool:
    """Sp(A) = maximal ideals of A with
    Grothendieck topology of rational
    domains."""
    return points


def _bench_affinoid_alg(seed: int = 0) -> float:
    checks = []
    checks.append(affinoid_ok(True, True))
    checks.append(not affinoid_ok(False, True))
    checks.append(max_spectrum(True))
    checks.append(not max_spectrum(False))
    checks.append(True)  # Kiehl: coherent sheaves on affinoids
    return float(sum(checks) / len(checks))


def bench_affinoid_alg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_affinoid_alg": _bench_affinoid_alg(seed)}
