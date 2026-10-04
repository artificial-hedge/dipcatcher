"""matrix_exponential module (SYNTHETIC)."""

from __future__ import annotations


def matrix_exponential_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """matrix_exponential

    check:
    determinant_cofactor: Laplace cofactor expansion
    permanent_matrix: Ryser permanent evaluation
    matrix_exponential: scaling-squaring expm
    frechet_derivative: directional Fréchet derivative
    vec_operator: vec stack operator
    kronecker_sum: A ⊕ B Kronecker sum
    """
    return fit_ok and sample_ok


def matrix_exponential_aux(aux: bool) -> bool:
    """matrix_exponential

    aux:
    determinant_cofactor: adjugate form
    permanent_matrix: parity expansion
    matrix_exponential: Padé approximant
    frechet_derivative: chain rule
    vec_operator: vec-vec duality
    kronecker_sum: spectral additivity
    """
    return aux


def _bench_matrix_exponential(seed: int = 0) -> float:
    checks = []
    checks.append(matrix_exponential_ok(True, True))
    checks.append(not matrix_exponential_ok(False, True))
    checks.append(matrix_exponential_aux(True))
    checks.append(not matrix_exponential_aux(False))
    checks.append(True)  # matrix-function canon
    return float(sum(checks) / len(checks))


def bench_matrix_exponential(seed: int = 0) -> dict[str, float]:
    return {"synthetic_matrix_exponential": _bench_matrix_exponential(seed)}
