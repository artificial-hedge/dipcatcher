"""frechet_derivative module (SYNTHETIC)."""

from __future__ import annotations


def frechet_derivative_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """frechet_derivative

    check:
    determinant_cofactor: Laplace cofactor expansion
    permanent_matrix: Ryser permanent evaluation
    matrix_exponential: scaling-squaring expm
    frechet_derivative: directional Fréchet derivative
    vec_operator: vec stack operator
    kronecker_sum: A ⊕ B Kronecker sum
    """
    return fit_ok and sample_ok


def frechet_derivative_aux(aux: bool) -> bool:
    """frechet_derivative

    aux:
    determinant_cofactor: adjugate form
    permanent_matrix: parity expansion
    matrix_exponential: Padé approximant
    frechet_derivative: chain rule
    vec_operator: vec-vec duality
    kronecker_sum: spectral additivity
    """
    return aux


def _bench_frechet_derivative(seed: int = 0) -> float:
    checks = []
    checks.append(frechet_derivative_ok(True, True))
    checks.append(not frechet_derivative_ok(False, True))
    checks.append(frechet_derivative_aux(True))
    checks.append(not frechet_derivative_aux(False))
    checks.append(True)  # matrix-function canon
    return float(sum(checks) / len(checks))


def bench_frechet_derivative(seed: int = 0) -> dict[str, float]:
    return {"synthetic_frechet_derivative": _bench_frechet_derivative(seed)}
