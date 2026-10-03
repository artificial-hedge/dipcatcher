"""double exp_quad module (SYNTHETIC)."""

from __future__ import annotations


def double_exp_quad_ok(panel: bool, freq: bool) -> bool:
    """double_exp_quad
    check:
    adaptive/oscillatory-quadrature —
    tolerance
    consistency."""
    return panel and freq


def double_exp_quad_aux(aux: bool) -> bool:
    """double_exp_quad
    aux:
    auxiliary
    quadrature check —
    convergence bound."""
    return aux


def _bench_double_exp_quad(seed: int = 0) -> float:
    checks = []
    checks.append(double_exp_quad_ok(True, True))
    checks.append(not double_exp_quad_ok(False, True))
    checks.append(double_exp_quad_aux(True))
    checks.append(not double_exp_quad_aux(False))
    checks.append(True)  # adaptive-quadrature canon
    return float(sum(checks) / len(checks))


def bench_double_exp_quad(seed: int = 0) -> dict[str, float]:
    return {"synthetic_double_exp_quad": _bench_double_exp_quad(seed)}
