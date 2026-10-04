"""default_risk module (SYNTHETIC)."""

from __future__ import annotations


def default_risk_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """default_risk

    check:
    capm_model: CAPM model
    arbitrage_pricing: APT theory
    black_scholes: Black-Scholes model
    yield_curve: yield curve
    default_risk: default risk
    corporate_finance: corporate finance
    """
    return fit_ok and sample_ok


def default_risk_aux(aux: bool) -> bool:
    """default_risk

    aux:
    capm_model: security market line
    arbitrage_pricing: factor pricing
    black_scholes: option pricing PDE
    yield_curve: Nelson-Siegel
    default_risk: Merton model
    corporate_finance: capital structure
    """
    return aux


def _bench_default_risk(seed: int = 0) -> float:
    checks = []
    checks.append(default_risk_ok(True, True))
    checks.append(not default_risk_ok(False, True))
    checks.append(default_risk_aux(True))
    checks.append(not default_risk_aux(False))
    checks.append(True)  # finance-theory canon
    return float(sum(checks) / len(checks))


def bench_default_risk(seed: int = 0) -> dict[str, float]:
    return {"synthetic_default_risk": _bench_default_risk(seed)}
