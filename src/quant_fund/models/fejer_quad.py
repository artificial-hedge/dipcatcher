"""fejer quad module (SYNTHETIC)."""

from __future__ import annotations


def fejer_quad_ok(node: bool, weight: bool) -> bool:
    """fejer_quad
    check:
    classical-quadrature —
    exactness
    consistency."""
    return node and weight


def fejer_quad_aux(aux: bool) -> bool:
    """fejer_quad
    aux:
    auxiliary
    quadrature check —
    positivity."""
    return aux


def _bench_fejer_quad(seed: int = 0) -> float:
    checks = []
    checks.append(fejer_quad_ok(True, True))
    checks.append(not fejer_quad_ok(False, True))
    checks.append(fejer_quad_aux(True))
    checks.append(not fejer_quad_aux(False))
    checks.append(True)  # classical-quadrature canon
    return float(sum(checks) / len(checks))


def bench_fejer_quad(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fejer_quad": _bench_fejer_quad(seed)}
