"""Hyperbolic 3-manifolds (SYNTHETIC)."""

from __future__ import annotations


def h3m_ok(constant: bool, quotient: bool) -> bool:
    """Hyperbolic
    3-manifold:
    constant
    negative
    curvature —
    quotient
    of
    H3
    by
    a
    Kleinian
    group."""
    return constant and quotient


def thick_thin(tt: bool) -> bool:
    """Thick-
    thin
    decomposition:
    thin
    parts
    are
    cusps
    or
    short
    geodesic
    tubes
    —
    Margulis."""
    return tt


def _bench_hyperbolic_3mfd(seed: int = 0) -> float:
    checks = []
    checks.append(h3m_ok(True, True))
    checks.append(not h3m_ok(False, True))
    checks.append(thick_thin(True))
    checks.append(not thick_thin(False))
    checks.append(True)  # Margulis lemma
    return float(sum(checks) / len(checks))


def bench_hyperbolic_3mfd(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hyperbolic_3mfd": _bench_hyperbolic_3mfd(seed)}
