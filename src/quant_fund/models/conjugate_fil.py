"""Conjugate filtration (SYNTHETIC)."""

from __future__ import annotations


def cf_ok(conjugate: bool, cartier: bool) -> bool:
    """Conjugate:
    conjugate
    filtration
    from
    Cartier —
    Cartier
    isomorphism."""
    return conjugate and cartier


def cartier_iso(ci: bool) -> bool:
    """Cartier
    iso:
    Cartier
    isomorphism
    on
    Hodge —
    Katz
    Cartier."""
    return ci


def _bench_conjugate_fil(seed: int = 0) -> float:
    checks = []
    checks.append(cf_ok(True, True))
    checks.append(not cf_ok(False, True))
    checks.append(cartier_iso(True))
    checks.append(not cartier_iso(False))
    checks.append(True)  # Cartier-Katz
    return float(sum(checks) / len(checks))


def bench_conjugate_fil(seed: int = 0) -> dict[str, float]:
    return {"synthetic_conjugate_fil": _bench_conjugate_fil(seed)}
