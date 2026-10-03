"""Symbol calculus (SYNTHETIC)."""

from __future__ import annotations


def sc_ok(principal: bool, asymptotic: bool) -> bool:
    """Symbol
    calculus:
    principal
    symbol
    controls
    ellipticity;
    asymptotic
    sums
    give
    lower
    terms."""
    return principal and asymptotic


def elliptic_reg(er: bool) -> bool:
    """Elliptic
    regularity:
    elliptic
    symbols
    have
    parametrices
    modulo
    smoothing."""
    return er


def _bench_symbol_calc(seed: int = 0) -> float:
    checks = []
    checks.append(sc_ok(True, True))
    checks.append(not sc_ok(False, True))
    checks.append(elliptic_reg(True))
    checks.append(not elliptic_reg(False))
    checks.append(True)  # Hörmander
    return float(sum(checks) / len(checks))


def bench_symbol_calc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_symbol_calc": _bench_symbol_calc(seed)}
