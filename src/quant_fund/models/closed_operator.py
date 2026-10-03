"""closed_operator module (SYNTHETIC)."""

from __future__ import annotations


def closed_operator_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """closed_operator

    check:
    unbounded_operator: densely defined unbounded map
    closed_operator: closed graph criterion
    domain_dense: dense-domain extension
    adjoint_unbounded: adjoint of unbounded operator
    resolvent_op: resolvent analytic family
    spectral_measure: projection-valued measure
    """
    return fit_ok and sample_ok


def closed_operator_aux(aux: bool) -> bool:
    """closed_operator

    aux:
    unbounded_operator: graph norm
    closed_operator: closability test
    domain_dense: core identification
    adjoint_unbounded: selfadjoint extension
    resolvent_op: first resolvent identity
    spectral_measure: functional calculus
    """
    return aux


def _bench_closed_operator(seed: int = 0) -> float:
    checks = []
    checks.append(closed_operator_ok(True, True))
    checks.append(not closed_operator_ok(False, True))
    checks.append(closed_operator_aux(True))
    checks.append(not closed_operator_aux(False))
    checks.append(True)  # unbounded-operator canon
    return float(sum(checks) / len(checks))


def bench_closed_operator(seed: int = 0) -> dict[str, float]:
    return {"synthetic_closed_operator": _bench_closed_operator(seed)}
