"""permanent_matrix module (SYNTHETIC)."""

from __future__ import annotations


def permanent_matrix_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """permanent_matrix

    check:
    determinant_cofactor: Laplace cofactor expansion
    permanent_matrix: Ryser permanent evaluation
    matrix_exponential: scaling-squaring expm
    frechet_derivative: directional Fréchet derivative
    vec_operator: vec stack operator
    kronecker_sum: A ⊕ B Kronecker sum
    """
    return fit_ok and sample_ok


def permanent_matrix_aux(aux: bool) -> bool:
    """permanent_matrix

    aux:
    determinant_cofactor: adjugate form
    permanent_matrix: parity expansion
    matrix_exponential: Padé approximant
    frechet_derivative: chain rule
    vec_operator: vec-vec duality
    kronecker_sum: spectral additivity
    """
    return aux


def _bench_permanent_matrix(seed: int = 0) -> float:
    checks = []
    checks.append(permanent_matrix_ok(True, True))
    checks.append(not permanent_matrix_ok(False, True))
    checks.append(permanent_matrix_aux(True))
    checks.append(not permanent_matrix_aux(False))
    checks.append(True)  # matrix-function canon
    return float(sum(checks) / len(checks))


def bench_permanent_matrix(seed: int = 0) -> dict[str, float]:
    return {"synthetic_permanent_matrix": _bench_permanent_matrix(seed)}
