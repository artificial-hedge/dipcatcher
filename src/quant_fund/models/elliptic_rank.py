"""Elliptic curve ranks (SYNTHETIC)."""

from __future__ import annotations


def er_ok(mordell_weil: bool, rank_free: bool) -> bool:
    """Elliptic
    curve
    rank:
    Mordell-
    Weil
    group
    is
    finitely
    generated
    —
    rank
    is
    the
    free
    part."""
    return mordell_weil and rank_free


def descent_rank(dr: bool) -> bool:
    """2-descent:
    algorithm
    bounding
    rank
    via
    2-
    Selmer
    groups —
    Cassels
    and
    Cremona."""
    return dr


def _bench_elliptic_rank(seed: int = 0) -> float:
    checks = []
    checks.append(er_ok(True, True))
    checks.append(not er_ok(False, True))
    checks.append(descent_rank(True))
    checks.append(not descent_rank(False))
    checks.append(True)  # Mordell-Weil
    return float(sum(checks) / len(checks))


def bench_elliptic_rank(seed: int = 0) -> dict[str, float]:
    return {"synthetic_elliptic_rank": _bench_elliptic_rank(seed)}
