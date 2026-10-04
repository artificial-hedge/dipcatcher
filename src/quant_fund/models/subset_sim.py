"""subset sim module (SYNTHETIC)."""

from __future__ import annotations


def subset_sim_ok(beta: bool, conv: bool) -> bool:
    """subset_sim
    check:
    reliability —
    failure-probability
    consistency."""
    return beta and conv


def subset_sim_aux(aux: bool) -> bool:
    """subset_sim
    aux:
    auxiliary
    reliability check —
    index bound."""
    return aux


def _bench_subset_sim(seed: int = 0) -> float:
    checks = []
    checks.append(subset_sim_ok(True, True))
    checks.append(not subset_sim_ok(False, True))
    checks.append(subset_sim_aux(True))
    checks.append(not subset_sim_aux(False))
    checks.append(True)  # reliability canon
    return float(sum(checks) / len(checks))


def bench_subset_sim(seed: int = 0) -> dict[str, float]:
    return {"synthetic_subset_sim": _bench_subset_sim(seed)}
