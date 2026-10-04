"""welding_technology module (SYNTHETIC)."""

from __future__ import annotations


def welding_technology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """welding_technology

    check:
    electrical_trades: electrical trades
    plumbing_hvac: plumbing and HVAC
    welding_technology: welding technology
    carpentry_trades: carpentry trades
    automotive_technology: automotive technology
    refrigeration_technology: refrigeration technology
    """
    return fit_ok and sample_ok


def welding_technology_aux(aux: bool) -> bool:
    """welding_technology

    aux:
    electrical_trades: circuits and wiring
    plumbing_hvac: pipes and climate
    welding_technology: joints and metals
    carpentry_trades: framing and joinery
    automotive_technology: engines and diagnostics
    refrigeration_technology: cooling and compressors
    """
    return aux


def _bench_welding_technology(seed: int = 0) -> float:
    checks = []
    checks.append(welding_technology_ok(True, True))
    checks.append(not welding_technology_ok(False, True))
    checks.append(welding_technology_aux(True))
    checks.append(not welding_technology_aux(False))
    checks.append(True)  # trades canon
    return float(sum(checks) / len(checks))


def bench_welding_technology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_welding_technology": _bench_welding_technology(seed)}
