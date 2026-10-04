"""overlapping_gens module (SYNTHETIC)."""

from __future__ import annotations


def overlapping_gens_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """overlapping_gens

    check:
    growth_theory: Solow growth model
    overlapping_gens: OLG model
    real_business: RBC model
    search_matching: search and matching
    mechanism_design: mechanism design
    auction_theory2: auction theory
    """
    return fit_ok and sample_ok


def overlapping_gens_aux(aux: bool) -> bool:
    """overlapping_gens

    aux:
    growth_theory: convergence
    overlapping_gens: Diamond model
    real_business: productivity shocks
    search_matching: Beveridge curve
    mechanism_design: incentive compatibility
    auction_theory2: revenue equivalence
    """
    return aux


def _bench_overlapping_gens(seed: int = 0) -> float:
    checks = []
    checks.append(overlapping_gens_ok(True, True))
    checks.append(not overlapping_gens_ok(False, True))
    checks.append(overlapping_gens_aux(True))
    checks.append(not overlapping_gens_aux(False))
    checks.append(True)  # economics canon
    return float(sum(checks) / len(checks))


def bench_overlapping_gens(seed: int = 0) -> dict[str, float]:
    return {"synthetic_overlapping_gens": _bench_overlapping_gens(seed)}
