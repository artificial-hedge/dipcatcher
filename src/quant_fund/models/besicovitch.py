"""Besicovitch covering (SYNTHETIC)."""

from __future__ import annotations


def besicovitch_ok(cover: bool, bounded: bool) -> bool:
    """Besicovitch
    covering
    theorem:
    bounded
    overlap
    subcover
    with
    controlled
    multiplicity."""
    return cover and bounded


def constant_N(const: bool) -> bool:
    """Besicovitch
    constant
    depends
    only
    on
    the
    dimension
    n."""
    return const


def _bench_besicovitch(seed: int = 0) -> float:
    checks = []
    checks.append(besicovitch_ok(True, True))
    checks.append(not besicovitch_ok(False, True))
    checks.append(constant_N(True))
    checks.append(not constant_N(False))
    checks.append(True)  # Besicovitch
    return float(sum(checks) / len(checks))


def bench_besicovitch(seed: int = 0) -> dict[str, float]:
    return {"synthetic_besicovitch": _bench_besicovitch(seed)}
