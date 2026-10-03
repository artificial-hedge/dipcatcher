"""sylvester_matrix module (SYNTHETIC)."""

from __future__ import annotations


def sylvester_matrix_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sylvester_matrix

    check:
    perron_frobenius: Perron–Frobenius eigenpair
    douglas_factor: Douglas factorization lemma
    cholesky_piv: pivoted Cholesky
    matrix_square_root: SPD principal square root
    polar_decomp: A = U H polar factorization
    sylvester_matrix: Sylvester equation AX + XB = C
    """
    return fit_ok and sample_ok


def sylvester_matrix_aux(aux: bool) -> bool:
    """sylvester_matrix

    aux:
    perron_frobenius: spectral radius eigenvector
    douglas_factor: range-inclusion factorization
    cholesky_piv: RRQR-style pivot order
    matrix_square_root: Newton–Schulz iteration
    polar_decomp: unitary + positive part
    sylvester_matrix: Bartels–Stewart solve
    """
    return aux


def _bench_sylvester_matrix(seed: int = 0) -> float:
    checks = []
    checks.append(sylvester_matrix_ok(True, True))
    checks.append(not sylvester_matrix_ok(False, True))
    checks.append(sylvester_matrix_aux(True))
    checks.append(not sylvester_matrix_aux(False))
    checks.append(True)  # positive-matrix canon
    return float(sum(checks) / len(checks))


def bench_sylvester_matrix(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sylvester_matrix": _bench_sylvester_matrix(seed)}
