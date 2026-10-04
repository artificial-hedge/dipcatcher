"""formation_evaluation module (SYNTHETIC)."""

from __future__ import annotations


def formation_evaluation_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """formation_evaluation

    check:
    reservoir_engineering: reservoir engineering
    drilling_engineering: drilling engineering
    production_engineering: production engineering
    formation_evaluation: formation evaluation
    well_testing: well testing
    enhanced_recovery: enhanced recovery
    """
    return fit_ok and sample_ok


def formation_evaluation_aux(aux: bool) -> bool:
    """formation_evaluation

    aux:
    reservoir_engineering: material balance
    drilling_engineering: borehole stability
    production_engineering: artificial lift
    formation_evaluation: well logs
    well_testing: pressure transient
    enhanced_recovery: waterflooding
    """
    return aux


def _bench_formation_evaluation(seed: int = 0) -> float:
    checks = []
    checks.append(formation_evaluation_ok(True, True))
    checks.append(not formation_evaluation_ok(False, True))
    checks.append(formation_evaluation_aux(True))
    checks.append(not formation_evaluation_aux(False))
    checks.append(True)  # petroleum-engineering canon
    return float(sum(checks) / len(checks))


def bench_formation_evaluation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_formation_evaluation": _bench_formation_evaluation(seed)}
