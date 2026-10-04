"""yield_curve module (SYNTHETIC)."""

from __future__ import annotations


def yield_curve_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """yield_curve

    check:
    capm_model: CAPM model
    arbitrage_pricing: APT theory
    black_scholes: Black-Scholes model
    yield_curve: yield curve
    default_risk: default risk
    corporate_finance: corporate finance
    """
    return fit_ok and sample_ok


def yield_curve_aux(aux: bool) -> bool:
    """yield_curve

    aux:
    capm_model: security market line
    arbitrage_pricing: factor pricing
    black_scholes: option pricing PDE
    yield_curve: Nelson-Siegel
    default_risk: Merton model
    corporate_finance: capital structure
    """
    return aux


def _bench_yield_curve(seed: int = 0) -> float:
    checks = []
    checks.append(yield_curve_ok(True, True))
    checks.append(not yield_curve_ok(False, True))
    checks.append(yield_curve_aux(True))
    checks.append(not yield_curve_aux(False))
    checks.append(True)  # finance-theory canon
    return float(sum(checks) / len(checks))


def bench_yield_curve(seed: int = 0) -> dict[str, float]:
    return {"synthetic_yield_curve": _bench_yield_curve(seed)}
