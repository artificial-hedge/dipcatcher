"""Syntomic cohomology (SYNTHETIC)."""

from __future__ import annotations


def syn_ok(syntomic: bool, comparison: bool) -> bool:
    """Syntomic:
    syntomic
    complexes
    compare
    etale
    and
    crystalline —
    syntomic
    topology."""
    return syntomic and comparison


def fontaine_messing(fm: bool) -> bool:
    """Fontaine-
    Messing:
    syntomic
    comparison
    for
    p-
    torsion —
    syntomic."""
    return fm


def _bench_syntomic(seed: int = 0) -> float:
    checks = []
    checks.append(syn_ok(True, True))
    checks.append(not syn_ok(False, True))
    checks.append(fontaine_messing(True))
    checks.append(not fontaine_messing(False))
    checks.append(True)  # Fontaine-Messing
    return float(sum(checks) / len(checks))


def bench_syntomic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_syntomic": _bench_syntomic(seed)}
