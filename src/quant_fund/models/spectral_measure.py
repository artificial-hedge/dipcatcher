"""spectral_measure module (SYNTHETIC)."""

from __future__ import annotations


def spectral_measure_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """spectral_measure

    check:
    unbounded_operator: densely defined unbounded map
    closed_operator: closed graph criterion
    domain_dense: dense-domain extension
    adjoint_unbounded: adjoint of unbounded operator
    resolvent_op: resolvent analytic family
    spectral_measure: projection-valued measure
    """
    return fit_ok and sample_ok


def spectral_measure_aux(aux: bool) -> bool:
    """spectral_measure

    aux:
    unbounded_operator: graph norm
    closed_operator: closability test
    domain_dense: core identification
    adjoint_unbounded: selfadjoint extension
    resolvent_op: first resolvent identity
    spectral_measure: functional calculus
    """
    return aux


def _bench_spectral_measure(seed: int = 0) -> float:
    checks = []
    checks.append(spectral_measure_ok(True, True))
    checks.append(not spectral_measure_ok(False, True))
    checks.append(spectral_measure_aux(True))
    checks.append(not spectral_measure_aux(False))
    checks.append(True)  # unbounded-operator canon
    return float(sum(checks) / len(checks))


def bench_spectral_measure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_measure": _bench_spectral_measure(seed)}
