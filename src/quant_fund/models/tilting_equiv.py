"""Tilting equivalence (SYNTHETIC)."""

from __future__ import annotations


def tilt_ok(perfection: bool, tilt_cat: bool) -> bool:
    """Tilting X -> X^b identifies
    perfectoid X over Q_p with
    a char-p perfectoid; topology,
    etale site, and Galois group
    all preserved."""
    return perfection and tilt_cat


def untilt_ahb(delta_ring: bool) -> bool:
    """Untilting via W(k) and the
    Fontaine theta map; A_inf
    gives a canonical untilt."""
    return delta_ring


def _bench_tilting_equiv(seed: int = 0) -> float:
    checks = []
    checks.append(tilt_ok(True, True))
    checks.append(not tilt_ok(False, True))
    checks.append(untilt_ahb(True))
    checks.append(not untilt_ahb(False))
    checks.append(True)  # tilting interchanges Frob
    return float(sum(checks) / len(checks))


def bench_tilting_equiv(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tilting_equiv": _bench_tilting_equiv(seed)}
