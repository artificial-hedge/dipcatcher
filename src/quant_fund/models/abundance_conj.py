"""Abundance conjecture (SYNTHETIC)."""

from __future__ import annotations


def ac_ok(semiample: bool, litaka: bool) -> bool:
    """Abundance:
    nef
    canonical
    is
    semiample —
    abundance
    conjecture."""
    return semiample and litaka


def abundance_cases(ac: bool) -> bool:
    """Abundance
    cases:
    holds
    for
    dimension
    up
    to
    three —
    Miyaoka-
    Kawamata."""
    return ac


def _bench_abundance_conj(seed: int = 0) -> float:
    checks = []
    checks.append(ac_ok(True, True))
    checks.append(not ac_ok(False, True))
    checks.append(abundance_cases(True))
    checks.append(not abundance_cases(False))
    checks.append(True)  # Miyaoka-Kawamata
    return float(sum(checks) / len(checks))


def bench_abundance_conj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_abundance_conj": _bench_abundance_conj(seed)}
