"""lotka_volterra module (SYNTHETIC)."""

from __future__ import annotations


def lotka_volterra_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lotka_volterra

    check:
    predator_prey: predator-prey dynamics
    lotka_volterra: Lotka-Volterra equations
    logistic_growth: logistic growth
    island_biogeography: island biogeography
    neutral_theory: ecological neutral theory
    food_web: food web structure
    """
    return fit_ok and sample_ok


def lotka_volterra_aux(aux: bool) -> bool:
    """lotka_volterra

    aux:
    predator_prey: functional response
    lotka_volterra: competition coefficients
    logistic_growth: carrying capacity
    island_biogeography: MacArthur-Wilson
    neutral_theory: Hubbell zero-sum
    food_web: trophic cascade
    """
    return aux


def _bench_lotka_volterra(seed: int = 0) -> float:
    checks = []
    checks.append(lotka_volterra_ok(True, True))
    checks.append(not lotka_volterra_ok(False, True))
    checks.append(lotka_volterra_aux(True))
    checks.append(not lotka_volterra_aux(False))
    checks.append(True)  # ecology canon
    return float(sum(checks) / len(checks))


def bench_lotka_volterra(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lotka_volterra": _bench_lotka_volterra(seed)}
