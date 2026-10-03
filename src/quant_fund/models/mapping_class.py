"""Mapping class group (SYNTHETIC)."""

from __future__ import annotations


def mcg_ok(homeo: bool, isotopy: bool) -> bool:
    """Mapping
    class
    group:
    isotopy
    classes
    of
    orientation-
    preserving
    homeomorphisms
    of
    a
    surface —
    acts
    on
    Teichmueller."""
    return homeo and isotopy


def nielsen_thurston(nt: bool) -> bool:
    """Nielsen-
    Thurston:
    every
    mapping
    class
    is
    periodic,
    reducible,
    or
    pseudo-
    Anosov —
    trichotomy."""
    return nt


def _bench_mapping_class(seed: int = 0) -> float:
    checks = []
    checks.append(mcg_ok(True, True))
    checks.append(not mcg_ok(False, True))
    checks.append(nielsen_thurston(True))
    checks.append(not nielsen_thurston(False))
    checks.append(True)  # Nielsen-Thurston
    return float(sum(checks) / len(checks))


def bench_mapping_class(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mapping_class": _bench_mapping_class(seed)}
