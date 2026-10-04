"""drilling_engineering module (SYNTHETIC)."""

from __future__ import annotations


def drilling_engineering_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """drilling_engineering

    check:
    reservoir_engineering: reservoir engineering
    drilling_engineering: drilling engineering
    production_engineering: production engineering
    formation_evaluation: formation evaluation
    well_testing: well testing
    enhanced_recovery: enhanced recovery
    """
    return fit_ok and sample_ok


def drilling_engineering_aux(aux: bool) -> bool:
    """drilling_engineering

    aux:
    reservoir_engineering: material balance
    drilling_engineering: borehole stability
    production_engineering: artificial lift
    formation_evaluation: well logs
    well_testing: pressure transient
    enhanced_recovery: waterflooding
    """
    return aux


def _bench_drilling_engineering(seed: int = 0) -> float:
    checks = []
    checks.append(drilling_engineering_ok(True, True))
    checks.append(not drilling_engineering_ok(False, True))
    checks.append(drilling_engineering_aux(True))
    checks.append(not drilling_engineering_aux(False))
    checks.append(True)  # petroleum-engineering canon
    return float(sum(checks) / len(checks))


def bench_drilling_engineering(seed: int = 0) -> dict[str, float]:
    return {"synthetic_drilling_engineering": _bench_drilling_engineering(seed)}
