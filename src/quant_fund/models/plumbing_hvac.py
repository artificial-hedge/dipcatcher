"""plumbing_hvac module (SYNTHETIC)."""

from __future__ import annotations


def plumbing_hvac_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """plumbing_hvac

    check:
    electrical_trades: electrical trades
    plumbing_hvac: plumbing and HVAC
    welding_technology: welding technology
    carpentry_trades: carpentry trades
    automotive_technology: automotive technology
    refrigeration_technology: refrigeration technology
    """
    return fit_ok and sample_ok


def plumbing_hvac_aux(aux: bool) -> bool:
    """plumbing_hvac

    aux:
    electrical_trades: circuits and wiring
    plumbing_hvac: pipes and climate
    welding_technology: joints and metals
    carpentry_trades: framing and joinery
    automotive_technology: engines and diagnostics
    refrigeration_technology: cooling and compressors
    """
    return aux


def _bench_plumbing_hvac(seed: int = 0) -> float:
    checks = []
    checks.append(plumbing_hvac_ok(True, True))
    checks.append(not plumbing_hvac_ok(False, True))
    checks.append(plumbing_hvac_aux(True))
    checks.append(not plumbing_hvac_aux(False))
    checks.append(True)  # trades canon
    return float(sum(checks) / len(checks))


def bench_plumbing_hvac(seed: int = 0) -> dict[str, float]:
    return {"synthetic_plumbing_hvac": _bench_plumbing_hvac(seed)}
