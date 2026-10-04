"""search_matching module (SYNTHETIC)."""

from __future__ import annotations


def search_matching_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """search_matching

    check:
    growth_theory: Solow growth model
    overlapping_gens: OLG model
    real_business: RBC model
    search_matching: search and matching
    mechanism_design: mechanism design
    auction_theory2: auction theory
    """
    return fit_ok and sample_ok


def search_matching_aux(aux: bool) -> bool:
    """search_matching

    aux:
    growth_theory: convergence
    overlapping_gens: Diamond model
    real_business: productivity shocks
    search_matching: Beveridge curve
    mechanism_design: incentive compatibility
    auction_theory2: revenue equivalence
    """
    return aux


def _bench_search_matching(seed: int = 0) -> float:
    checks = []
    checks.append(search_matching_ok(True, True))
    checks.append(not search_matching_ok(False, True))
    checks.append(search_matching_aux(True))
    checks.append(not search_matching_aux(False))
    checks.append(True)  # economics canon
    return float(sum(checks) / len(checks))


def bench_search_matching(seed: int = 0) -> dict[str, float]:
    return {"synthetic_search_matching": _bench_search_matching(seed)}
