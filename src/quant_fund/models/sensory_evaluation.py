"""sensory_evaluation module (SYNTHETIC)."""

from __future__ import annotations


def sensory_evaluation_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sensory_evaluation

    check:
    food_chemistry: food chemistry
    food_microbiology: food microbiology
    food_processing: food processing
    nutrition_science: nutrition science
    sensory_evaluation: sensory evaluation
    food_safety: food safety
    """
    return fit_ok and sample_ok


def sensory_evaluation_aux(aux: bool) -> bool:
    """sensory_evaluation

    aux:
    food_chemistry: Maillard reaction
    food_microbiology: fermentation
    food_processing: pasteurization
    nutrition_science: micronutrients
    sensory_evaluation: taste panels
    food_safety: HACCP
    """
    return aux


def _bench_sensory_evaluation(seed: int = 0) -> float:
    checks = []
    checks.append(sensory_evaluation_ok(True, True))
    checks.append(not sensory_evaluation_ok(False, True))
    checks.append(sensory_evaluation_aux(True))
    checks.append(not sensory_evaluation_aux(False))
    checks.append(True)  # food-science canon
    return float(sum(checks) / len(checks))


def bench_sensory_evaluation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sensory_evaluation": _bench_sensory_evaluation(seed)}
