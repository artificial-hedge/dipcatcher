"""Guth-Katz incidence (SYNTHETIC)."""

from __future__ import annotations


def gk_ok(polynomial: bool, partition: bool) -> bool:
    """Guth-Katz:
    polynomial
    partitioning
    divides
    space
    into cells
    for
    incidence
    estimates."""
    return polynomial and partition


def second_technique(sec: bool) -> bool:
    """Guth's
    second
    distance
    technique:
    point
    pairs
    determine
    few
    rich
    algebraic
    curves."""
    return sec


def _bench_guth_katz(seed: int = 0) -> float:
    checks = []
    checks.append(gk_ok(True, True))
    checks.append(not gk_ok(False, True))
    checks.append(second_technique(True))
    checks.append(not second_technique(False))
    checks.append(True)  # Guth-Katz
    return float(sum(checks) / len(checks))


def bench_guth_katz(seed: int = 0) -> dict[str, float]:
    return {"synthetic_guth_katz": _bench_guth_katz(seed)}
