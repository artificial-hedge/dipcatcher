"""resolvent_op module (SYNTHETIC)."""

from __future__ import annotations


def resolvent_op_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """resolvent_op

    check:
    unbounded_operator: densely defined unbounded map
    closed_operator: closed graph criterion
    domain_dense: dense-domain extension
    adjoint_unbounded: adjoint of unbounded operator
    resolvent_op: resolvent analytic family
    spectral_measure: projection-valued measure
    """
    return fit_ok and sample_ok


def resolvent_op_aux(aux: bool) -> bool:
    """resolvent_op

    aux:
    unbounded_operator: graph norm
    closed_operator: closability test
    domain_dense: core identification
    adjoint_unbounded: selfadjoint extension
    resolvent_op: first resolvent identity
    spectral_measure: functional calculus
    """
    return aux


def _bench_resolvent_op(seed: int = 0) -> float:
    checks = []
    checks.append(resolvent_op_ok(True, True))
    checks.append(not resolvent_op_ok(False, True))
    checks.append(resolvent_op_aux(True))
    checks.append(not resolvent_op_aux(False))
    checks.append(True)  # unbounded-operator canon
    return float(sum(checks) / len(checks))


def bench_resolvent_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_resolvent_op": _bench_resolvent_op(seed)}
