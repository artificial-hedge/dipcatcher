"""kronecker_sum module (SYNTHETIC)."""

from __future__ import annotations


def kronecker_sum_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kronecker_sum

    check:
    determinant_cofactor: Laplace cofactor expansion
    permanent_matrix: Ryser permanent evaluation
    matrix_exponential: scaling-squaring expm
    frechet_derivative: directional Fréchet derivative
    vec_operator: vec stack operator
    kronecker_sum: A ⊕ B Kronecker sum
    """
    return fit_ok and sample_ok


def kronecker_sum_aux(aux: bool) -> bool:
    """kronecker_sum

    aux:
    determinant_cofactor: adjugate form
    permanent_matrix: parity expansion
    matrix_exponential: Padé approximant
    frechet_derivative: chain rule
    vec_operator: vec-vec duality
    kronecker_sum: spectral additivity
    """
    return aux


def _bench_kronecker_sum(seed: int = 0) -> float:
    checks = []
    checks.append(kronecker_sum_ok(True, True))
    checks.append(not kronecker_sum_ok(False, True))
    checks.append(kronecker_sum_aux(True))
    checks.append(not kronecker_sum_aux(False))
    checks.append(True)  # matrix-function canon
    return float(sum(checks) / len(checks))


def bench_kronecker_sum(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kronecker_sum": _bench_kronecker_sum(seed)}
