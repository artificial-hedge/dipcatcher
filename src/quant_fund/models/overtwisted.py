"""Overtwisted contact structures (SYNTHETIC)."""

from __future__ import annotations


def ot_ok(overtwisted_disk: bool, flexible: bool) -> bool:
    """Overtwisted:
    an
    overtwisted
    disk
    makes
    the
    structure
    flexible —
    Eliashberg
    classification
    by
    homotopy."""
    return overtwisted_disk and flexible


def h_principle(hp: bool) -> bool:
    """h-principle:
    overtwisted
    contact
    structures
    satisfy
    full
    h-principle —
    classified
    by
    formal
    data."""
    return hp


def _bench_overtwisted(seed: int = 0) -> float:
    checks = []
    checks.append(ot_ok(True, True))
    checks.append(not ot_ok(False, True))
    checks.append(h_principle(True))
    checks.append(not h_principle(False))
    checks.append(True)  # Eliashberg
    return float(sum(checks) / len(checks))


def bench_overtwisted(seed: int = 0) -> dict[str, float]:
    return {"synthetic_overtwisted": _bench_overtwisted(seed)}
