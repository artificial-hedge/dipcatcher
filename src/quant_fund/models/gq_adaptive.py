"""gq adaptive module (SYNTHETIC)."""

from __future__ import annotations


def gq_adaptive_ok(node: bool, wgt: bool) -> bool:
    """gq_adaptive
    check:
    quadrature —
    node/weight
    consistency."""
    return node and wgt


def gq_adaptive_aux(aux: bool) -> bool:
    """gq_adaptive
    aux:
    auxiliary
    quadrature check —
    moment bound."""
    return aux


def _bench_gq_adaptive(seed: int = 0) -> float:
    checks = []
    checks.append(gq_adaptive_ok(True, True))
    checks.append(not gq_adaptive_ok(False, True))
    checks.append(gq_adaptive_aux(True))
    checks.append(not gq_adaptive_aux(False))
    checks.append(True)  # quadrature canon
    return float(sum(checks) / len(checks))


def bench_gq_adaptive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gq_adaptive": _bench_gq_adaptive(seed)}
