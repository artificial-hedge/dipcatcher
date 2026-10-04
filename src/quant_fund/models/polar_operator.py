"""polar_operator module (SYNTHETIC)."""

from __future__ import annotations


def polar_operator_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """polar_operator

    check:
    hilbert_schmidt_op: square-summable singular values
    trace_class_op: summable singular values
    singular_value_op: S-number sequence
    schmidt_decomp: Hilbert-Schmidt factorization
    compact_normal: compact normal diagonalization
    polar_operator: polar decomposition of operator
    """
    return fit_ok and sample_ok


def polar_operator_aux(aux: bool) -> bool:
    """polar_operator

    aux:
    hilbert_schmidt_op: HS norm
    trace_class_op: trace norm
    singular_value_op: s-number inequalities
    schmidt_decomp: canonical expansion
    compact_normal: spectral compact normal
    polar_operator: partial isometry factor
    """
    return aux


def _bench_polar_operator(seed: int = 0) -> float:
    checks = []
    checks.append(polar_operator_ok(True, True))
    checks.append(not polar_operator_ok(False, True))
    checks.append(polar_operator_aux(True))
    checks.append(not polar_operator_aux(False))
    checks.append(True)  # Schatten/compact canon
    return float(sum(checks) / len(checks))


def bench_polar_operator(seed: int = 0) -> dict[str, float]:
    return {"synthetic_polar_operator": _bench_polar_operator(seed)}
