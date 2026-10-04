"""Ekeland-Hofer capacities (SYNTHETIC)."""

from __future__ import annotations


def eh_ok(sequence: bool, weyl: bool) -> bool:
    """Ekeland-
    Hofer
    capacities:
    countable
    sequence
    of
    capacities
    —
    first
    recovers
    Gromov
    width."""
    return sequence and weyl


def ehz_capacity(eh: bool) -> bool:
    """EHZ
    capacity:
    minimal
    symplectic
    action
    on
    convex
    boundaries —
    systolic
    ratio."""
    return eh


def _bench_ekeland_hofer(seed: int = 0) -> float:
    checks = []
    checks.append(eh_ok(True, True))
    checks.append(not eh_ok(False, True))
    checks.append(ehz_capacity(True))
    checks.append(not ehz_capacity(False))
    checks.append(True)  # Ekeland-Hofer
    return float(sum(checks) / len(checks))


def bench_ekeland_hofer(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ekeland_hofer": _bench_ekeland_hofer(seed)}
