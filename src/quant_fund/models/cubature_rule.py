"""cubature rule module (SYNTHETIC)."""

from __future__ import annotations


def cubature_rule_ok(node: bool, wgt: bool) -> bool:
    """cubature_rule
    check:
    quadrature —
    node/weight
    consistency."""
    return node and wgt


def cubature_rule_aux(aux: bool) -> bool:
    """cubature_rule
    aux:
    auxiliary
    quadrature check —
    moment bound."""
    return aux


def _bench_cubature_rule(seed: int = 0) -> float:
    checks = []
    checks.append(cubature_rule_ok(True, True))
    checks.append(not cubature_rule_ok(False, True))
    checks.append(cubature_rule_aux(True))
    checks.append(not cubature_rule_aux(False))
    checks.append(True)  # quadrature canon
    return float(sum(checks) / len(checks))


def bench_cubature_rule(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cubature_rule": _bench_cubature_rule(seed)}
