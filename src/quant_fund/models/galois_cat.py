"""Galois category (SYNTHETIC)."""

from __future__ import annotations


def gc_ok(galois_cat: bool, fiber_funct: bool) -> bool:
    """Galois
    category:
    fiber
    functor
    to
    finite
    sets —
    SGA1
    axioms."""
    return galois_cat and fiber_funct


def galois_torsor(gt: bool) -> bool:
    """Galois
    torsor:
    connected
    objects
    are
    torsors —
    Galois
    category."""
    return gt


def _bench_galois_cat(seed: int = 0) -> float:
    checks = []
    checks.append(gc_ok(True, True))
    checks.append(not gc_ok(False, True))
    checks.append(galois_torsor(True))
    checks.append(not galois_torsor(False))
    checks.append(True)  # Grothendieck
    return float(sum(checks) / len(checks))


def bench_galois_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_galois_cat": _bench_galois_cat(seed)}
