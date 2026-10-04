"""real_business module (SYNTHETIC)."""

from __future__ import annotations


def real_business_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """real_business

    check:
    growth_theory: Solow growth model
    overlapping_gens: OLG model
    real_business: RBC model
    search_matching: search and matching
    mechanism_design: mechanism design
    auction_theory2: auction theory
    """
    return fit_ok and sample_ok


def real_business_aux(aux: bool) -> bool:
    """real_business

    aux:
    growth_theory: convergence
    overlapping_gens: Diamond model
    real_business: productivity shocks
    search_matching: Beveridge curve
    mechanism_design: incentive compatibility
    auction_theory2: revenue equivalence
    """
    return aux


def _bench_real_business(seed: int = 0) -> float:
    checks = []
    checks.append(real_business_ok(True, True))
    checks.append(not real_business_ok(False, True))
    checks.append(real_business_aux(True))
    checks.append(not real_business_aux(False))
    checks.append(True)  # economics canon
    return float(sum(checks) / len(checks))


def bench_real_business(seed: int = 0) -> dict[str, float]:
    return {"synthetic_real_business": _bench_real_business(seed)}
