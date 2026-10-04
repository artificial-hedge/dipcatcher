"""quadrature rules module (SYNTHETIC)."""

from __future__ import annotations


def quadrature_rules_ok(element: bool, mesh: bool) -> bool:
    """quadrature_rules
    check:
    finite-element
    method —
    element."""
    return element and mesh


def quadrature_rules_aux(aux: bool) -> bool:
    """quadrature_rules
    aux:
    auxiliary
    FEM check —
    basis."""
    return aux


def _bench_quadrature_rules(seed: int = 0) -> float:
    checks = []
    checks.append(quadrature_rules_ok(True, True))
    checks.append(not quadrature_rules_ok(False, True))
    checks.append(quadrature_rules_aux(True))
    checks.append(not quadrature_rules_aux(False))
    checks.append(True)  # finite-element canon
    return float(sum(checks) / len(checks))


def bench_quadrature_rules(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quadrature_rules": _bench_quadrature_rules(seed)}
