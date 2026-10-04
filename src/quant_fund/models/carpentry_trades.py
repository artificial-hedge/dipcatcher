"""carpentry_trades module (SYNTHETIC)."""

from __future__ import annotations


def carpentry_trades_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """carpentry_trades

    check:
    electrical_trades: electrical trades
    plumbing_hvac: plumbing and HVAC
    welding_technology: welding technology
    carpentry_trades: carpentry trades
    automotive_technology: automotive technology
    refrigeration_technology: refrigeration technology
    """
    return fit_ok and sample_ok


def carpentry_trades_aux(aux: bool) -> bool:
    """carpentry_trades

    aux:
    electrical_trades: circuits and wiring
    plumbing_hvac: pipes and climate
    welding_technology: joints and metals
    carpentry_trades: framing and joinery
    automotive_technology: engines and diagnostics
    refrigeration_technology: cooling and compressors
    """
    return aux


def _bench_carpentry_trades(seed: int = 0) -> float:
    checks = []
    checks.append(carpentry_trades_ok(True, True))
    checks.append(not carpentry_trades_ok(False, True))
    checks.append(carpentry_trades_aux(True))
    checks.append(not carpentry_trades_aux(False))
    checks.append(True)  # trades canon
    return float(sum(checks) / len(checks))


def bench_carpentry_trades(seed: int = 0) -> dict[str, float]:
    return {"synthetic_carpentry_trades": _bench_carpentry_trades(seed)}
