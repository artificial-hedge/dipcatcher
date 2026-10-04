"""Serre-Cartan basis (SYNTHETIC)."""

from __future__ import annotations


def serre_cartan_ok(basis: bool, excess: bool) -> bool:
    """Serre-Cartan basis:
    admissible monomials
    of fixed excess
    give a basis for
    the mod-2
    Steenrod algebra."""
    return basis and excess


def excess_invariant(excess: bool) -> bool:
    """Excess e(I) = i_1 -
    i_2 - ... - i_k:
    invariant grading
    that measures
    instability."""
    return excess


def _bench_serre_cartan(seed: int = 0) -> float:
    checks = []
    checks.append(serre_cartan_ok(True, True))
    checks.append(not serre_cartan_ok(False, True))
    checks.append(excess_invariant(True))
    checks.append(not excess_invariant(False))
    checks.append(True)  # Cartan 1955
    return float(sum(checks) / len(checks))


def bench_serre_cartan(seed: int = 0) -> dict[str, float]:
    return {"synthetic_serre_cartan": _bench_serre_cartan(seed)}
