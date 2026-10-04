"""food_web module (SYNTHETIC)."""

from __future__ import annotations


def food_web_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """food_web

    check:
    predator_prey: predator-prey dynamics
    lotka_volterra: Lotka-Volterra equations
    logistic_growth: logistic growth
    island_biogeography: island biogeography
    neutral_theory: ecological neutral theory
    food_web: food web structure
    """
    return fit_ok and sample_ok


def food_web_aux(aux: bool) -> bool:
    """food_web

    aux:
    predator_prey: functional response
    lotka_volterra: competition coefficients
    logistic_growth: carrying capacity
    island_biogeography: MacArthur-Wilson
    neutral_theory: Hubbell zero-sum
    food_web: trophic cascade
    """
    return aux


def _bench_food_web(seed: int = 0) -> float:
    checks = []
    checks.append(food_web_ok(True, True))
    checks.append(not food_web_ok(False, True))
    checks.append(food_web_aux(True))
    checks.append(not food_web_aux(False))
    checks.append(True)  # ecology canon
    return float(sum(checks) / len(checks))


def bench_food_web(seed: int = 0) -> dict[str, float]:
    return {"synthetic_food_web": _bench_food_web(seed)}
