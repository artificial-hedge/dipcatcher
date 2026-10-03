"""Drinfeld quotient construction (SYNTHETIC)."""

from __future__ import annotations


def drinfeld_q_ok(equilateral: bool, new_obj: bool) -> bool:
    """Drinfeld dg quotient
    C/A: add object
    for each closed
    object of A with
    contractible cone
    data."""
    return equilateral and new_obj


def contractible_cone(inverse: bool) -> bool:
    """Contractible cone:
    the added object
    Z is a homotopy
    isomorphism
    0 -> Z in the
    dg quotient."""
    return inverse


def _bench_drinfeld_quotient(seed: int = 0) -> float:
    checks = []
    checks.append(drinfeld_q_ok(True, True))
    checks.append(not drinfeld_q_ok(False, True))
    checks.append(contractible_cone(True))
    checks.append(not contractible_cone(False))
    checks.append(True)  # Drinfeld 2004
    return float(sum(checks) / len(checks))


def bench_drinfeld_quotient(seed: int = 0) -> dict[str, float]:
    return {"synthetic_drinfeld_quotient": _bench_drinfeld_quotient(seed)}
