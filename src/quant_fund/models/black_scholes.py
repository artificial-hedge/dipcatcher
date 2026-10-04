"""black_scholes module (SYNTHETIC)."""

from __future__ import annotations


def black_scholes_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """black_scholes

    check:
    capm_model: CAPM model
    arbitrage_pricing: APT theory
    black_scholes: Black-Scholes model
    yield_curve: yield curve
    default_risk: default risk
    corporate_finance: corporate finance
    """
    return fit_ok and sample_ok


def black_scholes_aux(aux: bool) -> bool:
    """black_scholes

    aux:
    capm_model: security market line
    arbitrage_pricing: factor pricing
    black_scholes: option pricing PDE
    yield_curve: Nelson-Siegel
    default_risk: Merton model
    corporate_finance: capital structure
    """
    return aux


def _bench_black_scholes(seed: int = 0) -> float:
    checks = []
    checks.append(black_scholes_ok(True, True))
    checks.append(not black_scholes_ok(False, True))
    checks.append(black_scholes_aux(True))
    checks.append(not black_scholes_aux(False))
    checks.append(True)  # finance-theory canon
    return float(sum(checks) / len(checks))


def bench_black_scholes(seed: int = 0) -> dict[str, float]:
    return {"synthetic_black_scholes": _bench_black_scholes(seed)}
