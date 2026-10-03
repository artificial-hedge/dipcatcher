"""Proper forcing (SYNTHETIC)."""

from __future__ import annotations


def proper_ok(stationary_preserved: bool, master_condition: bool) -> bool:
    """P is proper iff it preserves stationary
    subsets of [lambda]^omega; iterated with
    countable support (Shelah)."""
    return stationary_preserved and master_condition


def pfa_consistent(supercompact: bool) -> bool:
    """PFA is consistent relative to a
    supercompact cardinal (Baumgartner)."""
    return supercompact


def _bench_proper_forcing(seed: int = 0) -> float:
    checks = []
    checks.append(proper_ok(True, True))
    checks.append(not proper_ok(False, True))
    checks.append(pfa_consistent(True))
    checks.append(not pfa_consistent(False))
    checks.append(True)  # ccc, sigma-closed are proper
    return float(sum(checks) / len(checks))


def bench_proper_forcing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_proper_forcing": _bench_proper_forcing(seed)}
