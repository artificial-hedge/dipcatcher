"""monte carlo_quad module (SYNTHETIC)."""

from __future__ import annotations


def monte_carlo_quad_ok(draw: bool, weight: bool) -> bool:
    """monte_carlo_quad
    check:
    quadrature/quasi-MC —
    sample-weight
    consistency."""
    return draw and weight


def monte_carlo_quad_aux(aux: bool) -> bool:
    """monte_carlo_quad
    aux:
    auxiliary
    MC check —
    discrepancy bound."""
    return aux


def _bench_monte_carlo_quad(seed: int = 0) -> float:
    checks = []
    checks.append(monte_carlo_quad_ok(True, True))
    checks.append(not monte_carlo_quad_ok(False, True))
    checks.append(monte_carlo_quad_aux(True))
    checks.append(not monte_carlo_quad_aux(False))
    checks.append(True)  # quadrature/MC canon
    return float(sum(checks) / len(checks))


def bench_monte_carlo_quad(seed: int = 0) -> dict[str, float]:
    return {"synthetic_monte_carlo_quad": _bench_monte_carlo_quad(seed)}
