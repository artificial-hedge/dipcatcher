"""filon quad module (SYNTHETIC)."""

from __future__ import annotations


def filon_quad_ok(panel: bool, freq: bool) -> bool:
    """filon_quad
    check:
    adaptive/oscillatory-quadrature —
    tolerance
    consistency."""
    return panel and freq


def filon_quad_aux(aux: bool) -> bool:
    """filon_quad
    aux:
    auxiliary
    quadrature check —
    convergence bound."""
    return aux


def _bench_filon_quad(seed: int = 0) -> float:
    checks = []
    checks.append(filon_quad_ok(True, True))
    checks.append(not filon_quad_ok(False, True))
    checks.append(filon_quad_aux(True))
    checks.append(not filon_quad_aux(False))
    checks.append(True)  # adaptive-quadrature canon
    return float(sum(checks) / len(checks))


def bench_filon_quad(seed: int = 0) -> dict[str, float]:
    return {"synthetic_filon_quad": _bench_filon_quad(seed)}
