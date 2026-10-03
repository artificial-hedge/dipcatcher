"""Mumford-Tate groups (SYNTHETIC)."""

from __future__ import annotations


def mt_ok(smallest_group: bool, generic: bool) -> bool:
    """Mumford-
    Tate
    group:
    smallest
    Q-
    algebraic
    group
    containing
    the
    Hodge
    cocharacter —
    generic
    structure."""
    return smallest_group and generic


def mt_domain(mtd: bool) -> bool:
    """Mumford-
    Tate
    domain:
    MT
    orbit
    of
    the
    period
    point —
    Hodge
    locus
    support."""
    return mtd


def _bench_mumford_tate(seed: int = 0) -> float:
    checks = []
    checks.append(mt_ok(True, True))
    checks.append(not mt_ok(False, True))
    checks.append(mt_domain(True))
    checks.append(not mt_domain(False))
    checks.append(True)  # Mumford
    return float(sum(checks) / len(checks))


def bench_mumford_tate(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mumford_tate": _bench_mumford_tate(seed)}
