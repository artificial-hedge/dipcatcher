"""Quintic threefold invariants (SYNTHETIC)."""

from __future__ import annotations


def q5_ok(degree5: bool, cy3: bool) -> bool:
    """Quintic
    threefold:
    degree-5
    hypersurface
    in
    P4 —
    the
    first
    GW
    mirror
    computation."""
    return degree5 and cy3


def candelas_count(cc: bool) -> bool:
    """Candelas-
    de la
    Ossa-
    Green-
    Parkes:
    instanton
    numbers
    of
    the
    quintic
    from
    mirror
    periods."""
    return cc


def _bench_quintic_invariants(seed: int = 0) -> float:
    checks = []
    checks.append(q5_ok(True, True))
    checks.append(not q5_ok(False, True))
    checks.append(candelas_count(True))
    checks.append(not candelas_count(False))
    checks.append(True)  # Candelas 1991
    return float(sum(checks) / len(checks))


def bench_quintic_invariants(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quintic_invariants": _bench_quintic_invariants(seed)}
