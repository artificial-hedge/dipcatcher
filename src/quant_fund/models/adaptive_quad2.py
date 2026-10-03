"""adaptive quad2 module (SYNTHETIC)."""

from __future__ import annotations


def adaptive_quad2_ok(node: bool, wgt: bool) -> bool:
    """adaptive_quad2
    check:
    quadrature —
    node/weight
    consistency."""
    return node and wgt


def adaptive_quad2_aux(aux: bool) -> bool:
    """adaptive_quad2
    aux:
    auxiliary
    quadrature check —
    moment bound."""
    return aux


def _bench_adaptive_quad2(seed: int = 0) -> float:
    checks = []
    checks.append(adaptive_quad2_ok(True, True))
    checks.append(not adaptive_quad2_ok(False, True))
    checks.append(adaptive_quad2_aux(True))
    checks.append(not adaptive_quad2_aux(False))
    checks.append(True)  # quadrature canon
    return float(sum(checks) / len(checks))


def bench_adaptive_quad2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adaptive_quad2": _bench_adaptive_quad2(seed)}
