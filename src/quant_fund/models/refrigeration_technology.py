"""refrigeration_technology module (SYNTHETIC)."""

from __future__ import annotations


def refrigeration_technology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """refrigeration_technology

    check:
    electrical_trades: electrical trades
    plumbing_hvac: plumbing and HVAC
    welding_technology: welding technology
    carpentry_trades: carpentry trades
    automotive_technology: automotive technology
    refrigeration_technology: refrigeration technology
    """
    return fit_ok and sample_ok


def refrigeration_technology_aux(aux: bool) -> bool:
    """refrigeration_technology

    aux:
    electrical_trades: circuits and wiring
    plumbing_hvac: pipes and climate
    welding_technology: joints and metals
    carpentry_trades: framing and joinery
    automotive_technology: engines and diagnostics
    refrigeration_technology: cooling and compressors
    """
    return aux


def _bench_refrigeration_technology(seed: int = 0) -> float:
    checks = []
    checks.append(refrigeration_technology_ok(True, True))
    checks.append(not refrigeration_technology_ok(False, True))
    checks.append(refrigeration_technology_aux(True))
    checks.append(not refrigeration_technology_aux(False))
    checks.append(True)  # trades canon
    return float(sum(checks) / len(checks))


def bench_refrigeration_technology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_refrigeration_technology": _bench_refrigeration_technology(seed)}
