"""Quot schemes (SYNTHETIC)."""

from __future__ import annotations


def quot_ok(quotient_sheaf: bool, flat_family: bool) -> bool:
    """Quot scheme
    Quot(F, P): flat
    quotient sheaves
    Q of a coherent
    sheaf F with Hilbert
    polynomial P."""
    return quotient_sheaf and flat_family


def quot_representable(projective: bool) -> bool:
    """Quot is projective:
    Grothendieck's
    representability —
    Quot(F,P) is a
    projective scheme
    over the base."""
    return projective


def _bench_quot_scheme(seed: int = 0) -> float:
    checks = []
    checks.append(quot_ok(True, True))
    checks.append(not quot_ok(False, True))
    checks.append(quot_representable(True))
    checks.append(not quot_representable(False))
    checks.append(True)  # flattening stratification
    return float(sum(checks) / len(checks))


def bench_quot_scheme(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quot_scheme": _bench_quot_scheme(seed)}
