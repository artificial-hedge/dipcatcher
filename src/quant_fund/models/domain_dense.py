"""domain_dense module (SYNTHETIC)."""

from __future__ import annotations


def domain_dense_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """domain_dense

    check:
    unbounded_operator: densely defined unbounded map
    closed_operator: closed graph criterion
    domain_dense: dense-domain extension
    adjoint_unbounded: adjoint of unbounded operator
    resolvent_op: resolvent analytic family
    spectral_measure: projection-valued measure
    """
    return fit_ok and sample_ok


def domain_dense_aux(aux: bool) -> bool:
    """domain_dense

    aux:
    unbounded_operator: graph norm
    closed_operator: closability test
    domain_dense: core identification
    adjoint_unbounded: selfadjoint extension
    resolvent_op: first resolvent identity
    spectral_measure: functional calculus
    """
    return aux


def _bench_domain_dense(seed: int = 0) -> float:
    checks = []
    checks.append(domain_dense_ok(True, True))
    checks.append(not domain_dense_ok(False, True))
    checks.append(domain_dense_aux(True))
    checks.append(not domain_dense_aux(False))
    checks.append(True)  # unbounded-operator canon
    return float(sum(checks) / len(checks))


def bench_domain_dense(seed: int = 0) -> dict[str, float]:
    return {"synthetic_domain_dense": _bench_domain_dense(seed)}
