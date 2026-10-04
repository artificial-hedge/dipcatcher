"""well_testing module (SYNTHETIC)."""

from __future__ import annotations


def well_testing_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """well_testing

    check:
    reservoir_engineering: reservoir engineering
    drilling_engineering: drilling engineering
    production_engineering: production engineering
    formation_evaluation: formation evaluation
    well_testing: well testing
    enhanced_recovery: enhanced recovery
    """
    return fit_ok and sample_ok


def well_testing_aux(aux: bool) -> bool:
    """well_testing

    aux:
    reservoir_engineering: material balance
    drilling_engineering: borehole stability
    production_engineering: artificial lift
    formation_evaluation: well logs
    well_testing: pressure transient
    enhanced_recovery: waterflooding
    """
    return aux


def _bench_well_testing(seed: int = 0) -> float:
    checks = []
    checks.append(well_testing_ok(True, True))
    checks.append(not well_testing_ok(False, True))
    checks.append(well_testing_aux(True))
    checks.append(not well_testing_aux(False))
    checks.append(True)  # petroleum-engineering canon
    return float(sum(checks) / len(checks))


def bench_well_testing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_well_testing": _bench_well_testing(seed)}
