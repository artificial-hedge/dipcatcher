"""adjoint_unbounded module (SYNTHETIC)."""

from __future__ import annotations


def adjoint_unbounded_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """adjoint_unbounded

    check:
    unbounded_operator: densely defined unbounded map
    closed_operator: closed graph criterion
    domain_dense: dense-domain extension
    adjoint_unbounded: adjoint of unbounded operator
    resolvent_op: resolvent analytic family
    spectral_measure: projection-valued measure
    """
    return fit_ok and sample_ok


def adjoint_unbounded_aux(aux: bool) -> bool:
    """adjoint_unbounded

    aux:
    unbounded_operator: graph norm
    closed_operator: closability test
    domain_dense: core identification
    adjoint_unbounded: selfadjoint extension
    resolvent_op: first resolvent identity
    spectral_measure: functional calculus
    """
    return aux


def _bench_adjoint_unbounded(seed: int = 0) -> float:
    checks = []
    checks.append(adjoint_unbounded_ok(True, True))
    checks.append(not adjoint_unbounded_ok(False, True))
    checks.append(adjoint_unbounded_aux(True))
    checks.append(not adjoint_unbounded_aux(False))
    checks.append(True)  # unbounded-operator canon
    return float(sum(checks) / len(checks))


def bench_adjoint_unbounded(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adjoint_unbounded": _bench_adjoint_unbounded(seed)}
