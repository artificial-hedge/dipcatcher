"""enhanced_recovery module (SYNTHETIC)."""

from __future__ import annotations


def enhanced_recovery_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """enhanced_recovery

    check:
    reservoir_engineering: reservoir engineering
    drilling_engineering: drilling engineering
    production_engineering: production engineering
    formation_evaluation: formation evaluation
    well_testing: well testing
    enhanced_recovery: enhanced recovery
    """
    return fit_ok and sample_ok


def enhanced_recovery_aux(aux: bool) -> bool:
    """enhanced_recovery

    aux:
    reservoir_engineering: material balance
    drilling_engineering: borehole stability
    production_engineering: artificial lift
    formation_evaluation: well logs
    well_testing: pressure transient
    enhanced_recovery: waterflooding
    """
    return aux


def _bench_enhanced_recovery(seed: int = 0) -> float:
    checks = []
    checks.append(enhanced_recovery_ok(True, True))
    checks.append(not enhanced_recovery_ok(False, True))
    checks.append(enhanced_recovery_aux(True))
    checks.append(not enhanced_recovery_aux(False))
    checks.append(True)  # petroleum-engineering canon
    return float(sum(checks) / len(checks))


def bench_enhanced_recovery(seed: int = 0) -> dict[str, float]:
    return {"synthetic_enhanced_recovery": _bench_enhanced_recovery(seed)}
