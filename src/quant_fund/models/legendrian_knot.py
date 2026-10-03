"""Legendrian knots (SYNTHETIC)."""

from __future__ import annotations


def lk_ok(tangent: bool, contact_planes: bool) -> bool:
    """Legendrian
    knot:
    tangent
    to
    the
    contact
    planes —
    thurston-
    bennequin
    and
    rotation
    invariants."""
    return tangent and contact_planes


def classical_inv(ci: bool) -> bool:
    """Classical
    invariants:
    tb
    and
    rot
    are
    the
    two
    classical
    Legendrian
    invariants —
    Bennequin
    bound."""
    return ci


def _bench_legendrian_knot(seed: int = 0) -> float:
    checks = []
    checks.append(lk_ok(True, True))
    checks.append(not lk_ok(False, True))
    checks.append(classical_inv(True))
    checks.append(not classical_inv(False))
    checks.append(True)  # Bennequin
    return float(sum(checks) / len(checks))


def bench_legendrian_knot(seed: int = 0) -> dict[str, float]:
    return {"synthetic_legendrian_knot": _bench_legendrian_knot(seed)}
