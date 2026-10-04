"""island_biogeography module (SYNTHETIC)."""

from __future__ import annotations


def island_biogeography_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """island_biogeography

    check:
    predator_prey: predator-prey dynamics
    lotka_volterra: Lotka-Volterra equations
    logistic_growth: logistic growth
    island_biogeography: island biogeography
    neutral_theory: ecological neutral theory
    food_web: food web structure
    """
    return fit_ok and sample_ok


def island_biogeography_aux(aux: bool) -> bool:
    """island_biogeography

    aux:
    predator_prey: functional response
    lotka_volterra: competition coefficients
    logistic_growth: carrying capacity
    island_biogeography: MacArthur-Wilson
    neutral_theory: Hubbell zero-sum
    food_web: trophic cascade
    """
    return aux


def _bench_island_biogeography(seed: int = 0) -> float:
    checks = []
    checks.append(island_biogeography_ok(True, True))
    checks.append(not island_biogeography_ok(False, True))
    checks.append(island_biogeography_aux(True))
    checks.append(not island_biogeography_aux(False))
    checks.append(True)  # ecology canon
    return float(sum(checks) / len(checks))


def bench_island_biogeography(seed: int = 0) -> dict[str, float]:
    return {"synthetic_island_biogeography": _bench_island_biogeography(seed)}
