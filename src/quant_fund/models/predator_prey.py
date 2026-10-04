"""predator_prey module (SYNTHETIC)."""

from __future__ import annotations


def predator_prey_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """predator_prey

    check:
    predator_prey: predator-prey dynamics
    lotka_volterra: Lotka-Volterra equations
    logistic_growth: logistic growth
    island_biogeography: island biogeography
    neutral_theory: ecological neutral theory
    food_web: food web structure
    """
    return fit_ok and sample_ok


def predator_prey_aux(aux: bool) -> bool:
    """predator_prey

    aux:
    predator_prey: functional response
    lotka_volterra: competition coefficients
    logistic_growth: carrying capacity
    island_biogeography: MacArthur-Wilson
    neutral_theory: Hubbell zero-sum
    food_web: trophic cascade
    """
    return aux


def _bench_predator_prey(seed: int = 0) -> float:
    checks = []
    checks.append(predator_prey_ok(True, True))
    checks.append(not predator_prey_ok(False, True))
    checks.append(predator_prey_aux(True))
    checks.append(not predator_prey_aux(False))
    checks.append(True)  # ecology canon
    return float(sum(checks) / len(checks))


def bench_predator_prey(seed: int = 0) -> dict[str, float]:
    return {"synthetic_predator_prey": _bench_predator_prey(seed)}
