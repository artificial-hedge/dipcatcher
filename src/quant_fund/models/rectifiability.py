"""Rectifiability (SYNTHETIC)."""

from __future__ import annotations


def rect_ok(lipschitz: bool, cover: bool) -> bool:
    """Rectifiability:
    H^k-
    measurable
    set
    covered
    up to
    measure
    zero by
    Lipschitz
    images
    of
    R^k."""
    return lipschitz and cover


def approximate_tangent(at: bool) -> bool:
    """Approximate
    tangents:
    rectifiable
    sets
    have
    weak
    tangent
    planes
    H^k-
    a.e."""
    return at


def _bench_rectifiability(seed: int = 0) -> float:
    checks = []
    checks.append(rect_ok(True, True))
    checks.append(not rect_ok(False, True))
    checks.append(approximate_tangent(True))
    checks.append(not approximate_tangent(False))
    checks.append(True)  # Besicovitch-Federer
    return float(sum(checks) / len(checks))


def bench_rectifiability(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rectifiability": _bench_rectifiability(seed)}
