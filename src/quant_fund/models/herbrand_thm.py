"""Herbrand's theorem on finite term expansions (SYNTHETIC)."""

from __future__ import annotations


def tautology(vals: list[bool]) -> bool:
    return any(vals)


def herbrand_witness(exists_prop: list[bool]) -> bool:
    """Ex phi(x) provable iff phi(t1) or ... or phi(tn) tautology for
    some finite set of ground terms."""
    return tautology(exists_prop)


def _bench_herbrand_thm(seed: int = 0) -> float:
    checks = []
    # Ex(x=x) witnessed by any ground term
    checks.append(herbrand_witness([True, False]))
    # Ex(x!=x) has no witness
    checks.append(not herbrand_witness([False, False, False]))
    # single witness suffices
    checks.append(herbrand_witness([False, True]))
    # Herbrand disjunction over 2 terms
    checks.append(tautology([True, False]) != tautology([False, False]))
    return float(sum(checks) / len(checks))


def bench_herbrand_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_herbrand_thm": _bench_herbrand_thm(seed)}
